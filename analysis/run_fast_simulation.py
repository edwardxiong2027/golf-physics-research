#!/usr/bin/env python3
"""
Fast Simulation Runner - Optimized for speed while maintaining accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple
import os
import json
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure matplotlib
plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'figure.dpi': 150, 'savefig.dpi': 300})

# Constants
HOLE_RADIUS = 0.054
GRAVITY = 9.81

# PGA Data
PGA_MAKE_PCT = {2: 0.99, 3: 0.964, 4: 0.88, 5: 0.77, 6: 0.68, 7: 0.61, 8: 0.54, 9: 0.48,
                10: 0.40, 12: 0.33, 15: 0.23, 20: 0.15, 25: 0.10, 30: 0.07}

@dataclass
class GreenConditions:
    stimpmeter: float
    slope_percent: float
    slope_direction: float = np.pi/2
    @property
    def slope_angle(self): return np.arctan(self.slope_percent / 100)
    @property
    def friction_coefficient(self): return 0.65 / self.stimpmeter

def equations_of_motion(t, state, cond):
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 1e-6: return [0,0,0,0]
    theta, mu = cond.slope_angle, cond.friction_coefficient
    g_slope = GRAVITY * np.sin(theta)
    friction = mu * GRAVITY * np.cos(theta)
    return [vx, vy, 
            g_slope * np.cos(cond.slope_direction) - friction * vx/speed,
            g_slope * np.sin(cond.slope_direction) - friction * vy/speed]

def ball_stopped(t, state, cond): return np.sqrt(state[2]**2 + state[3]**2) - 0.01
ball_stopped.terminal = True
ball_stopped.direction = -1

def simulate_putt(distance, speed, aim_angle, cond):
    sol = solve_ivp(equations_of_motion, (0, 30), 
                    [0, 0, speed*np.cos(aim_angle), speed*np.sin(aim_angle)],
                    args=(cond,), events=ball_stopped, max_step=0.02)
    traj = sol.y.T
    hole_pos = np.array([distance, 0])
    min_dist = min(np.linalg.norm(traj[i,:2] - hole_pos) for i in range(len(traj)))
    made = False
    for i in range(len(traj)):
        d = np.linalg.norm(traj[i,:2] - hole_pos)
        v = np.sqrt(traj[i,2]**2 + traj[i,3]**2)
        if d <= HOLE_RADIUS and v <= 1.5 * (0.3 + 0.7*(1 - d/HOLE_RADIUS)):
            made = True; break
    return {'traj': traj, 'break': np.max(np.abs(traj[:,1])), 'made': made, 
            'min_dist': min_dist, 'final_pos': traj[-1,:2]}

def find_optimal_aim(distance, speed, cond):
    best = {'angle': 0, 'result': None, 'dist': float('inf')}
    for angle in np.linspace(-0.25, 0.25, 60):
        r = simulate_putt(distance, speed, angle, cond)
        if r['min_dist'] < best['dist']:
            best = {'angle': angle, 'result': r, 'dist': r['min_dist']}
    return best['angle'], best['result']

def calc_make_prob(distance, speed, cond, n=150):
    opt_aim, _ = find_optimal_aim(distance, speed, cond)
    makes = sum(1 for _ in range(n) 
                if simulate_putt(distance, 
                                 speed * (1 + np.random.normal(0, 0.04)),
                                 opt_aim + np.random.normal(0, 0.015), cond)['made'])
    return makes / n

def feet_to_m(ft): return ft * 0.3048
def m_to_in(m): return m * 39.37

print("="*60)
print("FAST GOLF PHYSICS SIMULATION")
print("="*60)

cond = GreenConditions(stimpmeter=12.0, slope_percent=2.0)
distance = feet_to_m(10)

# 1. Speed vs Break
print("\n1. Speed vs Break Analysis...")
speeds = np.linspace(0.9, 2.4, 20)
speed_break_data = []
for s in speeds:
    _, r = find_optimal_aim(distance, s, cond)
    speed_break_data.append({'speed_mps': s, 'break_inches': m_to_in(r['break']), 
                              'break_m': r['break'], 'made': r['made']})
    print(f"  {s:.2f} m/s -> {m_to_in(r['break']):.1f}\"")
speed_break_df = pd.DataFrame(speed_break_data)
speed_break_df.to_csv(f"{RESULTS_DIR}/speed_break_analysis.csv", index=False)

# 2. Optimal Speed
print("\n2. Optimal Speed Analysis...")
opt_speeds = np.linspace(1.2, 2.2, 15)
opt_data = []
for s in opt_speeds:
    prob = calc_make_prob(distance, s, cond, n=120)
    _, r = find_optimal_aim(distance, s, cond)
    past = max(0, r['final_pos'][0] - distance) * 39.37
    opt_data.append({'speed_mps': s, 'make_probability': prob, 'past_hole_inches': past})
    print(f"  {s:.2f} m/s -> {prob:.1%} make")
opt_df = pd.DataFrame(opt_data)
opt_df.to_csv(f"{RESULTS_DIR}/optimal_speed_analysis.csv", index=False)
opt_idx = opt_df['make_probability'].idxmax()
optimal_speed = opt_df.loc[opt_idx, 'speed_mps']
max_prob = opt_df.loc[opt_idx, 'make_probability']
print(f"\n  OPTIMAL: {optimal_speed:.2f} m/s at {max_prob:.1%}")

# 3. Multi-condition sweep (reduced)
print("\n3. Multi-Condition Sweep...")
sweep_data = []
for stimp in [10, 12, 13]:
    for slope in [1.0, 2.0, 3.0]:
        for dist_ft in [8, 10, 15, 20]:
            c = GreenConditions(stimpmeter=stimp, slope_percent=slope)
            d = feet_to_m(dist_ft)
            best_s, best_p = 1.5, 0
            for s in np.linspace(1.2, 2.0, 6):
                p = calc_make_prob(d, s, c, n=80)
                if p > best_p: best_s, best_p = s, p
            _, r = find_optimal_aim(d, best_s, c)
            sweep_data.append({'distance_ft': dist_ft, 'slope_percent': slope, 
                               'stimpmeter': stimp, 'optimal_speed_mps': best_s,
                               'max_make_probability': best_p, 'break_inches': m_to_in(r['break'])})
            print(f"  {dist_ft}ft, {slope}%, stimp{stimp} -> {best_s:.2f} m/s, {best_p:.1%}")
sweep_df = pd.DataFrame(sweep_data)
sweep_df.to_csv(f"{RESULTS_DIR}/multi_condition_sweep.csv", index=False)

# 4. Validation
print("\n4. Model Validation...")
val_cond = GreenConditions(stimpmeter=12.0, slope_percent=1.5)
val_data = []
for dist_ft, pga_pct in PGA_MAKE_PCT.items():
    if dist_ft > 20: continue
    d = feet_to_m(dist_ft)
    best_p = max(calc_make_prob(d, s, val_cond, n=100) for s in np.linspace(1.2, 2.0, 5))
    val_data.append({'distance_ft': dist_ft, 'pga_make_pct': pga_pct, 
                     'model_make_pct': best_p, 'error': abs(best_p - pga_pct)})
    print(f"  {dist_ft}ft: PGA={pga_pct:.1%}, Model={best_p:.1%}")
val_df = pd.DataFrame(val_data)
val_df.to_csv(f"{RESULTS_DIR}/pga_validation.csv", index=False)
corr = np.corrcoef(val_df['pga_make_pct'], val_df['model_make_pct'])[0,1]
mae = val_df['error'].mean()
print(f"\n  Correlation: {corr:.4f}, MAE: {mae:.3f}")

# 5. Generate Figures
print("\n5. Generating Figures...")

# Fig 1: Trajectories
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#228B22')
for speed, color in [(1.3, '#e41a1c'), (1.6, '#377eb8'), (1.9, '#4daf4a'), (2.2, '#984ea3')]:
    _, r = find_optimal_aim(distance, speed, cond)
    x, y = r['traj'][:,0]*100, r['traj'][:,1]*100
    ax.plot(x, y, color=color, lw=2.5, label=f'{speed:.1f} m/s ({m_to_in(r["break"]):.1f}")')
    ax.scatter([x[-1]], [y[-1]], s=80, c=color, edgecolors='white', zorder=5)
hole = plt.Circle((distance*100, 0), HOLE_RADIUS*100, color='black', zorder=4)
ax.add_patch(hole)
ax.scatter([0], [0], s=150, c='white', edgecolors='black', lw=2, label='Start', zorder=5)
ax.set_xlabel('Distance (cm)', color='white'); ax.set_ylabel('Lateral Break (cm)', color='white')
ax.set_title('Ball Trajectories: Effect of Speed on Break\n(10ft putt, 2% slope, Stimpmeter 12)', 
             color='white', fontweight='bold')
ax.legend(loc='upper left', facecolor='darkgreen', edgecolor='white', labelcolor='white')
ax.tick_params(colors='white'); ax.set_aspect('equal')
for spine in ax.spines.values(): spine.set_color('white')
plt.savefig(f"{FIGURES_DIR}/fig1_trajectory_comparison.png", facecolor='#228B22', dpi=300)
plt.close()
print("  Saved fig1_trajectory_comparison.png")

# Fig 2: Speed vs Break
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(speed_break_df['speed_mps'], speed_break_df['break_inches'], 'b-o', lw=2.5, ms=6)
# Fit
def power_law(x, k, a): return k / (x ** a)
valid = speed_break_df[speed_break_df['break_inches'] > 0]
popt, _ = curve_fit(power_law, valid['speed_mps'], valid['break_inches'], p0=[10,1], maxfev=5000)
x_fit = np.linspace(0.9, 2.4, 100)
ax.plot(x_fit, power_law(x_fit, *popt), 'r--', lw=2, label=f'Fit: B = {popt[0]:.1f}/v^{popt[1]:.2f}')
ax.set_xlabel('Initial Ball Speed (m/s)'); ax.set_ylabel('Total Break (inches)')
ax.set_title('Speed vs Break: Faster Putts Break Less', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(f"{FIGURES_DIR}/fig2_speed_vs_break.png", dpi=300)
plt.close()
print("  Saved fig2_speed_vs_break.png")

# Fig 3: Optimal Speed
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(opt_df['speed_mps'], opt_df['make_probability']*100, 'g-s', lw=2.5, ms=6)
ax.fill_between(opt_df['speed_mps'], opt_df['make_probability']*100, alpha=0.2, color='green')
ax.axvline(optimal_speed, color='red', ls='--', lw=2)
ax.scatter([optimal_speed], [max_prob*100], s=200, c='red', marker='*', zorder=5)
ax.annotate(f'Optimal: {optimal_speed:.2f} m/s\n({max_prob:.0%} make)', 
            xy=(optimal_speed, max_prob*100), xytext=(optimal_speed+0.2, max_prob*100-8),
            fontsize=11, color='red', arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Initial Ball Speed (m/s)'); ax.set_ylabel('Make Probability (%)')
ax.set_title('Finding the Optimal Putting Speed', fontweight='bold')
ax.grid(alpha=0.3)
plt.savefig(f"{FIGURES_DIR}/fig3_optimal_speed.png", dpi=300)
plt.close()
print("  Saved fig3_optimal_speed.png")

# Fig 4: Validation
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(val_df['distance_ft'], val_df['pga_make_pct']*100, s=100, c='navy', marker='o', 
           label='PGA Tour Data', zorder=3)
ax.plot(val_df['distance_ft'], val_df['model_make_pct']*100, 'r-s', lw=2, ms=6, 
        label='Physics Model')
ax.set_xlabel('Putt Distance (feet)'); ax.set_ylabel('Make Percentage (%)')
ax.set_title(f'Model Validation (r = {corr:.3f})', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 22); ax.set_ylim(0, 105)
plt.savefig(f"{FIGURES_DIR}/fig4_model_validation.png", dpi=300)
plt.close()
print("  Saved fig4_model_validation.png")

# Fig 5: Heatmap
fig, ax = plt.subplots(figsize=(9, 6))
pivot = sweep_df[sweep_df['stimpmeter']==12].pivot(index='slope_percent', columns='distance_ft', 
                                                    values='optimal_speed_mps')
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns))); ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.columns); ax.set_yticklabels([f'{s:.1f}' for s in pivot.index])
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center', fontsize=10)
ax.set_xlabel('Putt Distance (feet)'); ax.set_ylabel('Slope (%)')
ax.set_title('Optimal Speed (m/s) by Distance and Slope (Stimpmeter 12)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Optimal Speed (m/s)')
plt.savefig(f"{FIGURES_DIR}/fig5_optimal_speed_heatmap.png", dpi=300)
plt.close()
print("  Saved fig5_optimal_speed_heatmap.png")

# Fig 6: Force diagram
fig, ax = plt.subplots(figsize=(10, 7))
slope_pct, angle = 2.0, np.arctan(2.0/100)
ax.plot([0, 4], [0, -4*np.sin(angle)], 'g-', lw=4)
ax.fill_between([0, 4], [0, -4*np.sin(angle)], [-0.5, -0.5-4*np.sin(angle)], color='#228B22', alpha=0.3)
ball_x, ball_y = 2, -2*np.sin(angle) + 0.2
ball = plt.Circle((ball_x, ball_y), 0.15, color='white', ec='black', lw=2)
ax.add_patch(ball)
ax.arrow(ball_x, ball_y, 0, -0.6, head_width=0.08, head_length=0.05, fc='blue', ec='blue')
ax.text(ball_x+0.1, ball_y-0.35, 'mg', fontsize=14, color='blue', fontweight='bold')
ax.arrow(ball_x, ball_y, 0.5*np.sin(angle), 0.5*np.cos(angle), head_width=0.08, fc='green', ec='green')
ax.text(ball_x+0.5*np.sin(angle)+0.1, ball_y+0.5*np.cos(angle), 'N', fontsize=14, color='green', fontweight='bold')
ax.arrow(ball_x, ball_y, -0.4*np.cos(angle), 0.4*np.sin(angle), head_width=0.08, fc='red', ec='red')
ax.text(ball_x-0.4*np.cos(angle)-0.15, ball_y+0.4*np.sin(angle)+0.1, 'f', fontsize=14, color='red', fontweight='bold')
ax.arrow(ball_x, ball_y, 0.35*np.cos(angle), -0.35*np.sin(angle), head_width=0.08, fc='orange', ec='orange')
ax.text(ball_x+0.35*np.cos(angle)+0.1, ball_y-0.35*np.sin(angle)-0.1, 'mg sin θ', fontsize=12, color='orange', fontweight='bold')
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.8, 1); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('Free Body Diagram: Golf Ball on 2% Slope', fontsize=14, fontweight='bold')
legend_elements = [plt.Line2D([0],[0],color='blue',lw=3,label='Weight (mg)'),
                   plt.Line2D([0],[0],color='green',lw=3,label='Normal Force (N)'),
                   plt.Line2D([0],[0],color='red',lw=3,label='Friction (f = μN)'),
                   plt.Line2D([0],[0],color='orange',lw=3,label='Slope Component')]
ax.legend(handles=legend_elements, loc='upper right')
plt.savefig(f"{FIGURES_DIR}/fig6_force_diagram.png", dpi=300)
plt.close()
print("  Saved fig6_force_diagram.png")

# Save summary
summary = {
    'analysis_date': datetime.now().isoformat(),
    'conditions': {'stimpmeter': 12.0, 'slope_percent': 2.0, 'distance_ft': 10},
    'results': {
        'optimal_speed_mps': float(optimal_speed),
        'max_make_probability': float(max_prob),
        'speed_break_fit_k': float(popt[0]),
        'speed_break_fit_alpha': float(popt[1]),
        'validation_correlation': float(corr),
        'validation_mae': float(mae)
    }
}
with open(f"{RESULTS_DIR}/summary_results.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)
print(f"\nKey Results for 10ft putt, 2% slope, Stimpmeter 12:")
print(f"  Optimal Speed: {optimal_speed:.2f} m/s")
print(f"  Max Make Probability: {max_prob:.1%}")
print(f"  Speed-Break: B = {popt[0]:.1f}/v^{popt[1]:.2f}")
print(f"  Model-PGA Correlation: {corr:.3f}")
