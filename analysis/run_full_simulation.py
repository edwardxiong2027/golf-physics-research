#!/usr/bin/env python3
"""
Complete Simulation Runner for Golf Putting Physics Research

This script runs all simulations needed for the research paper:
1. Speed vs Break relationship
2. Optimal speed analysis
3. Multi-condition parameter sweep
4. Model validation against PGA data
5. Figure generation

Author: Jordan Xiong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, List, Dict
import os
import sys
import json
from datetime import datetime

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure matplotlib for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
GOLF_BALL_MASS = 0.04593  # kg
GOLF_BALL_RADIUS = 0.02135  # m
HOLE_RADIUS = 0.054  # m (4.25 inches diameter)
GRAVITY = 9.81  # m/s^2

# =============================================================================
# PGA DATA
# =============================================================================
PGA_MAKE_PERCENTAGE = {
    2: 0.990, 3: 0.964, 4: 0.880, 5: 0.770, 6: 0.680, 7: 0.610,
    8: 0.540, 9: 0.480, 10: 0.400, 11: 0.360, 12: 0.330, 13: 0.300,
    14: 0.270, 15: 0.230, 16: 0.210, 17: 0.195, 18: 0.180, 19: 0.168,
    20: 0.150, 22: 0.130, 25: 0.100, 30: 0.070
}

# =============================================================================
# PHYSICS MODEL
# =============================================================================

@dataclass
class GreenConditions:
    stimpmeter: float
    slope_percent: float
    slope_direction: float = np.pi/2  # Default: side slope
    
    @property
    def slope_angle(self) -> float:
        return np.arctan(self.slope_percent / 100)
    
    @property
    def friction_coefficient(self) -> float:
        return 0.65 / self.stimpmeter


@dataclass
class PuttResult:
    trajectory: np.ndarray
    times: np.ndarray
    final_position: Tuple[float, float]
    final_velocity: float
    total_break: float
    made: bool
    distance_from_hole: float


def equations_of_motion(t, state, conditions):
    x, y, vx, vy = state
    theta = conditions.slope_angle
    mu = conditions.friction_coefficient
    slope_dir = conditions.slope_direction
    
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 1e-6:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    vx_hat, vy_hat = vx / speed, vy / speed
    g_slope = GRAVITY * np.sin(theta)
    ax_gravity = g_slope * np.cos(slope_dir)
    ay_gravity = g_slope * np.sin(slope_dir)
    friction_decel = mu * GRAVITY * np.cos(theta)
    ax_friction = -friction_decel * vx_hat
    ay_friction = -friction_decel * vy_hat
    
    return np.array([vx, vy, ax_gravity + ax_friction, ay_gravity + ay_friction])


def ball_stopped(t, state, conditions):
    return np.sqrt(state[2]**2 + state[3]**2) - 0.01
ball_stopped.terminal = True
ball_stopped.direction = -1


def simulate_putt(distance, initial_speed, aim_angle, conditions, max_time=30.0):
    x0, y0 = 0.0, 0.0
    vx0 = initial_speed * np.cos(aim_angle)
    vy0 = initial_speed * np.sin(aim_angle)
    
    solution = solve_ivp(
        equations_of_motion, (0, max_time), [x0, y0, vx0, vy0],
        args=(conditions,), events=ball_stopped, dense_output=True, max_step=0.01
    )
    
    trajectory = solution.y.T
    final_pos = trajectory[-1, :2]
    final_vel = np.sqrt(trajectory[-1, 2]**2 + trajectory[-1, 3]**2)
    total_break = np.max(np.abs(trajectory[:, 1]))
    hole_pos = np.array([distance, 0.0])
    
    made = False
    min_dist = float('inf')
    
    for i in range(len(trajectory)):
        pos = trajectory[i, :2]
        vel = np.sqrt(trajectory[i, 2]**2 + trajectory[i, 3]**2)
        dist_to_hole = np.linalg.norm(pos - hole_pos)
        min_dist = min(min_dist, dist_to_hole)
        
        if dist_to_hole <= HOLE_RADIUS:
            center_factor = 1 - (dist_to_hole / HOLE_RADIUS)
            max_capture_vel = 1.5 * (0.3 + 0.7 * center_factor)
            if vel <= max_capture_vel:
                made = True
                break
    
    return PuttResult(trajectory, solution.t, tuple(final_pos), final_vel, 
                      total_break, made, min_dist)


def find_optimal_aim(distance, speed, conditions, angle_range=(-0.3, 0.3), num_angles=80):
    angles = np.linspace(angle_range[0], angle_range[1], num_angles)
    best_angle, best_result, min_dist = 0.0, None, float('inf')
    
    for angle in angles:
        result = simulate_putt(distance, speed, angle, conditions)
        if result.distance_from_hole < min_dist:
            min_dist = result.distance_from_hole
            best_angle = angle
            best_result = result
    
    return best_angle, best_result


def calculate_make_probability(distance, speed, conditions, 
                               aim_uncertainty=0.015, speed_uncertainty=0.04,
                               num_samples=300):
    optimal_aim, _ = find_optimal_aim(distance, speed, conditions)
    makes = 0
    
    for _ in range(num_samples):
        actual_aim = optimal_aim + np.random.normal(0, aim_uncertainty)
        actual_speed = speed * (1 + np.random.normal(0, speed_uncertainty))
        if actual_speed > 0:
            result = simulate_putt(distance, actual_speed, actual_aim, conditions)
            if result.made:
                makes += 1
    
    return makes / num_samples


def feet_to_meters(ft): return ft * 0.3048
def meters_to_feet(m): return m / 0.3048
def meters_to_inches(m): return m * 39.37

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def run_speed_break_analysis(distance_ft, conditions, speed_range=(0.8, 2.8), num_speeds=25):
    """Analyze how ball speed affects break magnitude."""
    print(f"\n{'='*60}")
    print(f"Speed vs Break Analysis: {distance_ft}ft putt, {conditions.slope_percent}% slope")
    print(f"{'='*60}")
    
    distance = feet_to_meters(distance_ft)
    speeds = np.linspace(speed_range[0], speed_range[1], num_speeds)
    results = []
    
    for speed in speeds:
        _, result = find_optimal_aim(distance, speed, conditions)
        if result:
            break_inches = meters_to_inches(result.total_break)
            results.append({
                'speed_mps': speed,
                'break_inches': break_inches,
                'break_m': result.total_break,
                'made': result.made
            })
            print(f"  Speed: {speed:.2f} m/s -> Break: {break_inches:.1f} inches")
    
    return pd.DataFrame(results)


def run_optimal_speed_analysis(distance_ft, conditions, speed_range=(0.9, 2.5), num_speeds=20):
    """Find optimal putting speed that maximizes make probability."""
    print(f"\n{'='*60}")
    print(f"Optimal Speed Analysis: {distance_ft}ft putt")
    print(f"Stimpmeter: {conditions.stimpmeter}, Slope: {conditions.slope_percent}%")
    print(f"{'='*60}")
    
    distance = feet_to_meters(distance_ft)
    speeds = np.linspace(speed_range[0], speed_range[1], num_speeds)
    results = []
    
    for speed in speeds:
        prob = calculate_make_probability(distance, speed, conditions, num_samples=250)
        
        # Calculate how far ball would roll past on flat surface
        _, result = find_optimal_aim(distance, speed, conditions)
        past_distance = max(0, result.final_position[0] - distance)
        past_inches = meters_to_inches(past_distance)
        
        results.append({
            'speed_mps': speed,
            'make_probability': prob,
            'past_hole_inches': past_inches
        })
        print(f"  Speed: {speed:.2f} m/s -> Make: {prob:.1%}, Past: {past_inches:.1f}\"")
    
    df = pd.DataFrame(results)
    optimal_idx = df['make_probability'].idxmax()
    optimal_speed = df.loc[optimal_idx, 'speed_mps']
    max_prob = df.loc[optimal_idx, 'make_probability']
    
    print(f"\n  OPTIMAL: {optimal_speed:.2f} m/s with {max_prob:.1%} make rate")
    
    return df, optimal_speed, max_prob


def run_multi_condition_sweep():
    """Run parameter sweep across multiple conditions."""
    print(f"\n{'='*60}")
    print("Multi-Condition Parameter Sweep")
    print(f"{'='*60}")
    
    distances = [8, 10, 15, 20]  # feet
    slopes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # percent
    stimpmeters = [10, 11, 12, 13]
    
    results = []
    
    for stimp in stimpmeters:
        for slope in slopes:
            conditions = GreenConditions(stimpmeter=stimp, slope_percent=slope)
            
            for dist_ft in distances:
                distance = feet_to_meters(dist_ft)
                
                # Quick optimal speed search
                speeds = np.linspace(1.0, 2.2, 10)
                best_speed, best_prob = 0, 0
                
                for speed in speeds:
                    prob = calculate_make_probability(distance, speed, conditions, num_samples=150)
                    if prob > best_prob:
                        best_prob = prob
                        best_speed = speed
                
                # Calculate break at optimal speed
                _, result = find_optimal_aim(distance, best_speed, conditions)
                break_inches = meters_to_inches(result.total_break) if result else 0
                
                results.append({
                    'distance_ft': dist_ft,
                    'slope_percent': slope,
                    'stimpmeter': stimp,
                    'optimal_speed_mps': best_speed,
                    'max_make_probability': best_prob,
                    'break_inches': break_inches
                })
                
                print(f"  Dist: {dist_ft}ft, Slope: {slope}%, Stimp: {stimp} -> "
                      f"Opt: {best_speed:.2f} m/s, Make: {best_prob:.1%}")
    
    return pd.DataFrame(results)


def validate_against_pga():
    """Compare model predictions to PGA Tour statistics."""
    print(f"\n{'='*60}")
    print("Model Validation Against PGA Tour Data")
    print(f"{'='*60}")
    
    # Use average tour conditions: stimpmeter 12, slight slope
    conditions = GreenConditions(stimpmeter=12.0, slope_percent=1.5)
    
    results = []
    for dist_ft, pga_pct in PGA_MAKE_PERCENTAGE.items():
        if dist_ft > 25:  # Skip very long putts for speed
            continue
            
        distance = feet_to_meters(dist_ft)
        
        # Find make probability at optimal speed
        speeds = np.linspace(1.0, 2.2, 8)
        best_prob = 0
        
        for speed in speeds:
            prob = calculate_make_probability(distance, speed, conditions, num_samples=200)
            best_prob = max(best_prob, prob)
        
        error = best_prob - pga_pct
        pct_error = abs(error / pga_pct) * 100 if pga_pct > 0 else 0
        
        results.append({
            'distance_ft': dist_ft,
            'pga_make_pct': pga_pct,
            'model_make_pct': best_prob,
            'absolute_error': error,
            'percent_error': pct_error
        })
        
        print(f"  {dist_ft}ft: PGA={pga_pct:.1%}, Model={best_prob:.1%}, Error={pct_error:.1f}%")
    
    df = pd.DataFrame(results)
    mae = df['absolute_error'].abs().mean()
    correlation = np.corrcoef(df['pga_make_pct'], df['model_make_pct'])[0,1]
    
    print(f"\n  Mean Absolute Error: {mae:.3f}")
    print(f"  Correlation: {correlation:.4f}")
    
    return df, mae, correlation

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figure1_trajectory(conditions, distance_ft=10):
    """Figure 1: Ball trajectories at different speeds."""
    print("\nGenerating Figure 1: Trajectory Comparison...")
    
    distance = feet_to_meters(distance_ft)
    speeds = [1.2, 1.6, 2.0, 2.4]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#228B22')
    
    for speed, color in zip(speeds, colors):
        aim, result = find_optimal_aim(distance, speed, conditions)
        if result:
            x = result.trajectory[:, 0] * 100  # cm
            y = result.trajectory[:, 1] * 100
            label = f'{speed:.1f} m/s (break: {meters_to_inches(result.total_break):.1f}")'
            ax.plot(x, y, color=color, linewidth=2.5, label=label)
            ax.scatter([x[-1]], [y[-1]], s=80, c=color, edgecolors='white', linewidths=1, zorder=5)
    
    # Draw hole
    hole_x, hole_y = distance * 100, 0
    hole = plt.Circle((hole_x, hole_y), HOLE_RADIUS * 100, color='black', zorder=4)
    ax.add_patch(hole)
    
    # Start position
    ax.scatter([0], [0], s=150, c='white', edgecolors='black', linewidths=2, 
               label='Start', zorder=5)
    
    ax.set_xlabel('Distance (cm)', fontsize=12, color='white')
    ax.set_ylabel('Lateral Break (cm)', fontsize=12, color='white')
    ax.set_title(f'Ball Trajectories: Effect of Speed on Break\n({distance_ft}ft putt, {conditions.slope_percent}% side slope, Stimpmeter {conditions.stimpmeter})',
                 fontsize=13, color='white', fontweight='bold')
    ax.legend(loc='upper left', facecolor='darkgreen', edgecolor='white', 
              labelcolor='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    filepath = os.path.join(FIGURES_DIR, "fig1_trajectory_comparison.png")
    fig.savefig(filepath, facecolor='#228B22', edgecolor='none', dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure2_speed_break(speed_break_df):
    """Figure 2: Speed vs Break relationship."""
    print("\nGenerating Figure 2: Speed vs Break...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(speed_break_df['speed_mps'], speed_break_df['break_inches'], 
            'b-', linewidth=2.5, marker='o', markersize=6)
    
    # Fit power law
    def power_law(x, k, alpha):
        return k / (x ** alpha)
    
    popt, _ = curve_fit(power_law, speed_break_df['speed_mps'], 
                        speed_break_df['break_inches'], p0=[10, 1])
    
    x_fit = np.linspace(0.8, 2.8, 100)
    y_fit = power_law(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
            label=f'Fit: B = {popt[0]:.2f}/v^{popt[1]:.2f}')
    
    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Total Break (inches)', fontsize=12)
    ax.set_title('Speed vs Break: Faster Putts Break Less', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Faster = Less Break', xy=(2.4, speed_break_df['break_inches'].iloc[-3]),
                xytext=(2.0, speed_break_df['break_inches'].iloc[5]),
                fontsize=10, color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
    
    filepath = os.path.join(FIGURES_DIR, "fig2_speed_vs_break.png")
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return popt  # Return fit parameters


def generate_figure3_optimal_speed(optimal_df, optimal_speed, max_prob):
    """Figure 3: Make probability vs speed."""
    print("\nGenerating Figure 3: Optimal Speed...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(optimal_df['speed_mps'], optimal_df['make_probability'] * 100,
            'g-', linewidth=2.5, marker='s', markersize=6)
    ax.fill_between(optimal_df['speed_mps'], optimal_df['make_probability'] * 100,
                    alpha=0.2, color='green')
    
    # Mark optimal
    ax.axvline(optimal_speed, color='red', linestyle='--', linewidth=2)
    ax.scatter([optimal_speed], [max_prob * 100], s=200, c='red', marker='*', zorder=5)
    ax.annotate(f'Optimal: {optimal_speed:.2f} m/s\n({max_prob:.0%} make rate)',
                xy=(optimal_speed, max_prob * 100),
                xytext=(optimal_speed + 0.3, max_prob * 100 - 10),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Make Probability (%)', fontsize=12)
    ax.set_title('Finding the Optimal Putting Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(FIGURES_DIR, "fig3_optimal_speed.png")
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure4_validation(validation_df):
    """Figure 4: Model vs PGA data comparison."""
    print("\nGenerating Figure 4: Model Validation...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(validation_df['distance_ft'], validation_df['pga_make_pct'] * 100,
               s=100, c='navy', marker='o', label='PGA Tour Data', zorder=3)
    ax.plot(validation_df['distance_ft'], validation_df['model_make_pct'] * 100,
            'r-', linewidth=2, marker='s', markersize=6, label='Physics Model')
    
    ax.set_xlabel('Putt Distance (feet)', fontsize=12)
    ax.set_ylabel('Make Percentage (%)', fontsize=12)
    ax.set_title('Model Validation: Physics Predictions vs PGA Tour Statistics',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 105)
    
    filepath = os.path.join(FIGURES_DIR, "fig4_model_validation.png")
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure5_heatmap(sweep_df):
    """Figure 5: Optimal speed heatmap."""
    print("\nGenerating Figure 5: Optimal Speed Heatmap...")
    
    # Filter for stimpmeter 12
    df_stimp12 = sweep_df[sweep_df['stimpmeter'] == 12]
    
    pivot = df_stimp12.pivot(index='slope_percent', columns='distance_ft', 
                             values='optimal_speed_mps')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f'{d}' for d in pivot.columns])
    ax.set_yticklabels([f'{s:.1f}' for s in pivot.index])
    
    # Add text
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color='black', fontsize=10)
    
    ax.set_xlabel('Putt Distance (feet)', fontsize=12)
    ax.set_ylabel('Slope (%)', fontsize=12)
    ax.set_title('Optimal Putting Speed (m/s) by Distance and Slope\n(Stimpmeter 12)',
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Optimal Speed (m/s)', fontsize=11)
    
    filepath = os.path.join(FIGURES_DIR, "fig5_optimal_speed_heatmap.png")
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure6_force_diagram():
    """Figure 6: Free body diagram."""
    print("\nGenerating Figure 6: Force Diagram...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    slope_percent = 2.0
    angle = np.arctan(slope_percent / 100)
    angle_deg = np.degrees(angle)
    
    # Draw surface
    ax.plot([0, 4], [0, -4 * np.sin(angle)], 'g-', linewidth=4, label='Green Surface')
    ax.fill_between([0, 4], [0, -4 * np.sin(angle)], [-0.5, -0.5 - 4*np.sin(angle)],
                    color='#228B22', alpha=0.3)
    
    # Ball position
    ball_x = 2
    ball_y = -2 * np.sin(angle) + 0.2
    ball = plt.Circle((ball_x, ball_y), 0.15, color='white', edgecolor='black', linewidth=2)
    ax.add_patch(ball)
    
    # Force arrows (scaled for visibility)
    arrow_props = dict(head_width=0.08, head_length=0.05, fc='blue', ec='blue')
    
    # Gravity
    ax.arrow(ball_x, ball_y, 0, -0.6, **dict(arrow_props, fc='blue', ec='blue'))
    ax.text(ball_x + 0.1, ball_y - 0.35, 'mg', fontsize=14, color='blue', fontweight='bold')
    
    # Normal force
    nx = 0.5 * np.sin(angle)
    ny = 0.5 * np.cos(angle)
    ax.arrow(ball_x, ball_y, nx, ny, **dict(arrow_props, fc='green', ec='green'))
    ax.text(ball_x + nx + 0.1, ball_y + ny, 'N', fontsize=14, color='green', fontweight='bold')
    
    # Friction
    fx = -0.4 * np.cos(angle)
    fy = 0.4 * np.sin(angle)
    ax.arrow(ball_x, ball_y, fx, fy, **dict(arrow_props, fc='red', ec='red'))
    ax.text(ball_x + fx - 0.15, ball_y + fy + 0.1, 'f', fontsize=14, color='red', fontweight='bold')
    
    # Slope component
    sx = 0.35 * np.cos(angle)
    sy = -0.35 * np.sin(angle)
    ax.arrow(ball_x, ball_y, sx, sy, **dict(arrow_props, fc='orange', ec='orange'))
    ax.text(ball_x + sx + 0.1, ball_y + sy - 0.1, 'mg sin θ', fontsize=12, color='orange', fontweight='bold')
    
    # Angle arc
    arc_angles = np.linspace(-angle, 0, 20)
    arc_x = 0.6 * np.cos(arc_angles)
    arc_y = 0.6 * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'k-', linewidth=1.5)
    ax.text(0.45, -0.08, f'θ', fontsize=12)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.8, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f'Free Body Diagram: Golf Ball on {slope_percent}% Slope',
                 fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Weight (mg)'),
        plt.Line2D([0], [0], color='green', linewidth=3, label='Normal Force (N)'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Friction (f = μN)'),
        plt.Line2D([0], [0], color='orange', linewidth=3, label='Slope Component (mg sin θ)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    filepath = os.path.join(FIGURES_DIR, "fig6_force_diagram.png")
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("GOLF PUTTING PHYSICS RESEARCH - FULL SIMULATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Standard conditions for main analysis
    conditions = GreenConditions(stimpmeter=12.0, slope_percent=2.0)
    
    # 1. Speed vs Break Analysis
    speed_break_df = run_speed_break_analysis(10, conditions)
    speed_break_df.to_csv(os.path.join(RESULTS_DIR, "speed_break_analysis.csv"), index=False)
    
    # 2. Optimal Speed Analysis
    optimal_df, optimal_speed, max_prob = run_optimal_speed_analysis(10, conditions)
    optimal_df.to_csv(os.path.join(RESULTS_DIR, "optimal_speed_analysis.csv"), index=False)
    
    # 3. Multi-condition sweep
    sweep_df = run_multi_condition_sweep()
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "multi_condition_sweep.csv"), index=False)
    
    # 4. Validation
    validation_df, mae, correlation = validate_against_pga()
    validation_df.to_csv(os.path.join(RESULTS_DIR, "pga_validation.csv"), index=False)
    
    # 5. Generate all figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    generate_figure1_trajectory(conditions)
    fit_params = generate_figure2_speed_break(speed_break_df)
    generate_figure3_optimal_speed(optimal_df, optimal_speed, max_prob)
    generate_figure4_validation(validation_df)
    generate_figure5_heatmap(sweep_df)
    generate_figure6_force_diagram()
    
    # 6. Save summary results
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'conditions': {
            'stimpmeter': conditions.stimpmeter,
            'slope_percent': conditions.slope_percent
        },
        'results': {
            'optimal_speed_mps': float(optimal_speed),
            'max_make_probability': float(max_prob),
            'speed_break_fit_k': float(fit_params[0]),
            'speed_break_fit_alpha': float(fit_params[1]),
            'validation_mae': float(mae),
            'validation_correlation': float(correlation)
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "summary_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  Optimal Speed (10ft, 2% slope, stimp 12): {optimal_speed:.2f} m/s")
    print(f"  Maximum Make Probability: {max_prob:.1%}")
    print(f"  Speed-Break Relationship: B = {fit_params[0]:.2f}/v^{fit_params[1]:.2f}")
    print(f"  Model-PGA Correlation: {correlation:.4f}")
    print(f"\nOutput files saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    return summary


if __name__ == "__main__":
    summary = main()
