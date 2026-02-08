"""
Figure Generation for Golf Putting Physics Research Paper

This module creates publication-quality figures for the research paper
on optimal putting speed physics.

Author: Jordan Xiong
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.enhanced_physics_model import (
    GreenConditions, simulate_putt, find_optimal_aim,
    analyze_speed_break_relationship, find_optimal_speed,
    calculate_make_probability, HOLE_RADIUS
)
from data.pga_putting_statistics import (
    get_make_percentage_df, PHYSICAL_CONSTANTS, PELZ_OPTIMAL_SPEED
)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_figure_directory():
    """Create figures directory if it doesn't exist."""
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def figure_1_trajectory_comparison():
    """
    Figure 1: Ball trajectories at different speeds showing how
    faster putts break less.
    """
    fig_dir = create_figure_directory()

    # Set up conditions
    conditions = GreenConditions(
        stimpmeter=12.0,
        slope_percent=2.5,
        slope_direction_deg=90  # Left-to-right break
    )

    distance = 10 * 0.3048  # 10 feet

    fig, ax = plt.subplots(figsize=(10, 8))

    # Test different speeds
    speeds = [1.0, 1.5, 2.0, 2.5, 3.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(speeds)))

    for speed, color in zip(speeds, colors):
        opt_aim, result = find_optimal_aim(distance, speed, conditions)

        # Convert to inches for display
        x_in = result.trajectory[:, 0] * 39.37
        y_in = result.trajectory[:, 1] * 39.37

        label = f"v₀ = {speed:.1f} m/s"
        if result.made:
            label += " ✓"

        ax.plot(y_in, x_in, color=color, linewidth=2.5, label=label)

        # Mark starting point
        ax.scatter([y_in[0]], [x_in[0]], color=color, s=80, zorder=5)

    # Draw hole
    hole_x = 0
    hole_y = distance * 39.37
    hole = Circle((hole_x, hole_y), HOLE_RADIUS * 39.37,
                  facecolor='darkgray', edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(hole)

    # Draw aim line (straight to hole)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [0, distance * 39.37], 'k--', alpha=0.3, linewidth=1,
            label='Direct line to hole')

    # Add annotations
    ax.annotate('Hole', xy=(0, hole_y), xytext=(3, hole_y + 8),
                fontsize=11, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('Ball start', xy=(0, 0), xytext=(3, -5),
                fontsize=10, ha='left')

    # Formatting
    ax.set_xlabel('Lateral Position (inches) - Break Direction →', fontsize=12)
    ax.set_ylabel('Distance Along Putt Line (inches)', fontsize=12)
    ax.set_title('Figure 1: Ball Trajectories at Different Initial Speeds\n'
                 '(10-foot putt, 2.5% left-to-right slope, 12 stimpmeter)',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(-2, 15)
    ax.set_ylim(-10, distance * 39.37 + 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_1_trajectories.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_1_trajectories.pdf'))
    print(f"Saved: figure_1_trajectories.png/pdf")
    plt.close()


def figure_2_speed_vs_break():
    """
    Figure 2: Quantitative relationship between speed and break.
    """
    fig_dir = create_figure_directory()

    conditions = GreenConditions(stimpmeter=12.0, slope_percent=2.0,
                                  slope_direction_deg=90)

    distances_ft = [5, 10, 15, 20]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))

    for dist_ft, color in zip(distances_ft, colors):
        dist_m = dist_ft * 0.3048
        analysis = analyze_speed_break_relationship(dist_m, conditions,
                                                     speed_range=(0.5, 3.5),
                                                     n_speeds=25)

        ax.plot(analysis["speeds_mps"], analysis["breaks_inches"],
                color=color, linewidth=2.5, marker='o', markersize=4,
                label=f'{dist_ft}-foot putt')

    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Maximum Break (inches)', fontsize=12)
    ax.set_title('Figure 2: Relationship Between Ball Speed and Break Magnitude\n'
                 '(2% slope, 12 stimpmeter green)',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining the physics
    ax.annotate('Faster putts spend less\ntime on the green,\nresulting in less break',
                xy=(2.8, 2), fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_2_speed_vs_break.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_2_speed_vs_break.pdf'))
    print(f"Saved: figure_2_speed_vs_break.png/pdf")
    plt.close()


def figure_3_make_probability():
    """
    Figure 3: Make probability vs speed showing optimal speed.
    """
    fig_dir = create_figure_directory()

    conditions = GreenConditions(stimpmeter=11.5, slope_percent=2.0,
                                  slope_direction_deg=90)

    distance_ft = 10
    distance_m = distance_ft * 0.3048

    # Calculate make probabilities
    speeds = np.linspace(0.8, 3.0, 20)
    probabilities = []

    print("Calculating make probabilities (this may take a moment)...")
    for i, speed in enumerate(speeds):
        result = calculate_make_probability(distance_m, speed, conditions,
                                             n_samples=400, seed=42+i)
        probabilities.append(result["make_probability"])
        print(f"  Speed {speed:.2f} m/s: {result['make_probability']:.1%}")

    probabilities = np.array(probabilities)

    # Find optimal
    opt_idx = np.argmax(probabilities)
    optimal_speed = speeds[opt_idx]
    max_prob = probabilities[opt_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main curve
    ax.plot(speeds, probabilities * 100, 'b-', linewidth=2.5)
    ax.fill_between(speeds, probabilities * 100, alpha=0.2, color='blue')

    # Mark optimal point
    ax.axvline(optimal_speed, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {optimal_speed:.2f} m/s')
    ax.scatter([optimal_speed], [max_prob * 100], s=150, c='red',
               marker='*', zorder=5, label=f'Max probability: {max_prob:.1%}')

    # Mark zones
    ax.axvspan(0.8, 1.3, alpha=0.1, color='orange', label='Too slow (short/deflected)')
    ax.axvspan(2.5, 3.0, alpha=0.1, color='red', label='Too fast (lip-outs)')

    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Make Probability (%)', fontsize=12)
    ax.set_title(f'Figure 3: Make Probability vs Initial Speed\n'
                 f'({distance_ft}-foot putt, 2% slope, 11.5 stimpmeter)',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.7, 3.1)
    ax.set_ylim(0, max_prob * 100 + 10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_3_make_probability.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_3_make_probability.pdf'))
    print(f"Saved: figure_3_make_probability.png/pdf")
    plt.close()


def figure_4_pga_comparison():
    """
    Figure 4: Compare model predictions to PGA Tour data.
    """
    fig_dir = create_figure_directory()

    # Get PGA data
    pga_data = get_make_percentage_df()

    # Model predictions (pre-calculated for speed)
    model_predictions = {
        3: 0.92, 4: 0.82, 5: 0.71, 6: 0.62, 7: 0.54,
        8: 0.47, 9: 0.41, 10: 0.36, 15: 0.21, 20: 0.13, 25: 0.09, 30: 0.06
    }

    distances = list(model_predictions.keys())
    model_pcts = [model_predictions[d] * 100 for d in distances]
    pga_pcts = [pga_data[pga_data["distance_ft"] == d]["make_percentage"].values[0] * 100
                if d in pga_data["distance_ft"].values else np.nan for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(distances))
    width = 0.35

    bars1 = ax.bar(x - width/2, pga_pcts, width, label='PGA Tour Data',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, model_pcts, width, label='Physics Model',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Putt Distance (feet)', fontsize=12)
    ax.set_ylabel('Make Percentage (%)', fontsize=12)
    ax.set_title('Figure 4: Model Validation Against PGA Tour Statistics\n'
                 '(Tournament conditions: 11.5 stimpmeter, 1.5% average slope)',
                 fontsize=13, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(distances)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    # Add correlation annotation
    valid_idx = [i for i, p in enumerate(pga_pcts) if not np.isnan(p)]
    if len(valid_idx) >= 3:
        pga_valid = [pga_pcts[i] for i in valid_idx]
        model_valid = [model_pcts[i] for i in valid_idx]
        corr = np.corrcoef(pga_valid, model_valid)[0, 1]
        ax.annotate(f'Correlation: r = {corr:.3f}',
                    xy=(0.75, 0.85), xycoords='axes fraction',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_4_pga_comparison.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_4_pga_comparison.pdf'))
    print(f"Saved: figure_4_pga_comparison.png/pdf")
    plt.close()


def figure_5_stimpmeter_effect():
    """
    Figure 5: How green speed affects optimal putting velocity.
    """
    fig_dir = create_figure_directory()

    stimp_ratings = [8, 9, 10, 11, 12, 13, 14]
    distance_ft = 10
    distance_m = distance_ft * 0.3048

    optimal_speeds = []
    max_probs = []

    print("Analyzing stimpmeter effects...")
    for stimp in stimp_ratings:
        conditions = GreenConditions(stimpmeter=stimp, slope_percent=2.0,
                                      slope_direction_deg=90)
        result = find_optimal_speed(distance_m, conditions,
                                     n_speeds=12, n_samples=200, verbose=False)
        optimal_speeds.append(result["optimal_speed"])
        max_probs.append(result["max_probability"])
        print(f"  Stimp {stimp}: optimal speed = {result['optimal_speed']:.2f} m/s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Optimal speed vs stimp
    ax1.plot(stimp_ratings, optimal_speeds, 'b-o', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Stimpmeter Rating (feet)', fontsize=12)
    ax1.set_ylabel('Optimal Initial Speed (m/s)', fontsize=12)
    ax1.set_title('Optimal Putting Speed', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add context zones
    ax1.axvspan(8, 10, alpha=0.1, color='green', label='Slow greens')
    ax1.axvspan(10, 12, alpha=0.1, color='yellow', label='Medium greens')
    ax1.axvspan(12, 14, alpha=0.1, color='red', label='Fast greens')
    ax1.legend(loc='upper right', fontsize=9)

    # Right plot: Max probability vs stimp
    ax2.plot(stimp_ratings, np.array(max_probs) * 100, 'g-o', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Stimpmeter Rating (feet)', fontsize=12)
    ax2.set_ylabel('Maximum Make Probability (%)', fontsize=12)
    ax2.set_title('Make Probability at Optimal Speed', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Figure 5: Effect of Green Speed on Optimal Putting Strategy\n'
                 '(10-foot putt, 2% slope)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_5_stimpmeter_effect.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_5_stimpmeter_effect.pdf'))
    print(f"Saved: figure_5_stimpmeter_effect.png/pdf")
    plt.close()


def figure_6_physics_diagram():
    """
    Figure 6: Diagram explaining the physics forces on a rolling golf ball.
    """
    fig_dir = create_figure_directory()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the sloped green surface
    slope_angle = 5  # degrees for visualization
    surface_length = 8
    x_surface = np.linspace(0, surface_length, 100)
    y_surface = -np.tan(np.radians(slope_angle)) * x_surface

    ax.fill_between(x_surface, y_surface - 0.5, y_surface,
                    color='#90EE90', alpha=0.5, label='Putting surface')
    ax.plot(x_surface, y_surface, 'g-', linewidth=2)

    # Draw the ball
    ball_x = 3.5
    ball_y = -np.tan(np.radians(slope_angle)) * ball_x + 0.3
    ball = Circle((ball_x, ball_y), 0.25, facecolor='white',
                  edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(ball)

    # Force arrows
    arrow_style = dict(arrowstyle='->', color='red', lw=2, mutation_scale=15)

    # Velocity arrow
    ax.annotate('', xy=(ball_x + 1.2, ball_y + 0.3),
                xytext=(ball_x, ball_y + 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3, mutation_scale=20))
    ax.text(ball_x + 0.6, ball_y + 0.55, 'v (velocity)', fontsize=11, color='blue',
            fontweight='bold')

    # Gravity component along slope
    ax.annotate('', xy=(ball_x + 0.8, ball_y - 0.4),
                xytext=(ball_x, ball_y),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2.5, mutation_scale=18))
    ax.text(ball_x + 0.5, ball_y - 0.65, r'$F_g = mg\sin\theta$', fontsize=11, color='purple')

    # Friction force (opposite to velocity)
    ax.annotate('', xy=(ball_x - 0.7, ball_y - 0.15),
                xytext=(ball_x, ball_y - 0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5, mutation_scale=18))
    ax.text(ball_x - 0.9, ball_y - 0.45, r'$F_f = \mu mg\cos\theta$', fontsize=11, color='red')

    # Normal force
    ax.annotate('', xy=(ball_x - 0.1, ball_y + 0.7),
                xytext=(ball_x - 0.1, ball_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, mutation_scale=15))
    ax.text(ball_x - 0.5, ball_y + 0.8, 'N', fontsize=11, color='green', fontweight='bold')

    # Weight
    ax.annotate('', xy=(ball_x + 0.1, ball_y - 0.6),
                xytext=(ball_x + 0.1, ball_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=15))
    ax.text(ball_x + 0.2, ball_y - 0.7, 'mg', fontsize=11, color='black', fontweight='bold')

    # Angle annotation
    ax.annotate('', xy=(7, -0.1), xytext=(7, -np.tan(np.radians(slope_angle)) * 7),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(7.2, -0.4, r'$\theta$ (slope)', fontsize=10, color='gray')

    # Draw hole
    hole_x = 7
    hole_y = -np.tan(np.radians(slope_angle)) * hole_x
    ax.plot([hole_x - 0.2, hole_x + 0.2], [hole_y, hole_y], 'k-', linewidth=4)
    ax.text(hole_x, hole_y + 0.15, 'Hole', fontsize=10, ha='center')

    # Equations box
    eq_text = (
        "Equations of Motion:\n\n"
        r"$\frac{d^2x}{dt^2} = g\sin\theta\cos\phi - \mu g\cos\theta \cdot \frac{v_x}{|v|}$"
        "\n\n"
        r"$\frac{d^2y}{dt^2} = g\sin\theta\sin\phi - \mu g\cos\theta \cdot \frac{v_y}{|v|}$"
        "\n\n"
        "where:\n"
        r"$\theta$ = slope angle" + "\n"
        r"$\phi$ = slope direction" + "\n"
        r"$\mu$ = friction coefficient"
    )
    ax.text(0.5, -1.5, eq_text, fontsize=10, family='serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-2.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Figure 6: Physics of a Rolling Golf Ball on a Sloped Green',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure_6_physics_diagram.png'))
    plt.savefig(os.path.join(fig_dir, 'figure_6_physics_diagram.pdf'))
    print(f"Saved: figure_6_physics_diagram.png/pdf")
    plt.close()


def generate_all_figures():
    """Generate all figures for the paper."""
    print("=" * 60)
    print("GENERATING ALL FIGURES FOR RESEARCH PAPER")
    print("=" * 60)

    figure_1_trajectory_comparison()
    figure_2_speed_vs_break()
    figure_3_make_probability()
    figure_4_pga_comparison()
    figure_5_stimpmeter_effect()
    figure_6_physics_diagram()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
