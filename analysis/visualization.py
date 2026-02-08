"""
Visualization Module for Golf Putting Research

Creates publication-quality figures for the research paper.

Author: Jordan Xiong
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Tuple, List
import os

# Configure matplotlib for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for consistency
COLORS = {
    'primary': '#1f77b4',    # Blue
    'secondary': '#ff7f0e',  # Orange
    'success': '#2ca02c',    # Green
    'danger': '#d62728',     # Red
    'neutral': '#7f7f7f',    # Gray
}


def plot_ball_trajectory(trajectory: np.ndarray, 
                         hole_position: Tuple[float, float],
                         output_path: Optional[str] = None,
                         title: str = "Putt Trajectory") -> None:
    """
    Plot a single putt trajectory showing ball path and break.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Shape (N, 4) with columns [x, y, vx, vy]
    hole_position : tuple
        (x, y) position of hole in meters
    output_path : str, optional
        Path to save figure
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = trajectory[:, 0] * 100  # Convert to cm
    y = trajectory[:, 1] * 100
    
    hole_x = hole_position[0] * 100
    hole_y = hole_position[1] * 100
    hole_radius = 5.4  # 10.8 cm diameter / 2
    
    # Draw green background
    ax.set_facecolor('#228B22')
    
    # Draw trajectory
    ax.plot(x, y, color='white', linewidth=2, label='Ball Path')
    
    # Mark start and end
    ax.scatter([x[0]], [y[0]], s=200, c='white', marker='o', 
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax.scatter([x[-1]], [y[-1]], s=200, c='yellow', marker='o',
               edgecolors='black', linewidths=2, label='End', zorder=5)
    
    # Draw hole
    hole_circle = plt.Circle((hole_x, hole_y), hole_radius, 
                              color='black', fill=True, zorder=4)
    ax.add_patch(hole_circle)
    
    # Draw aim line (straight line from start to hole)
    ax.plot([x[0], hole_x], [y[0], hole_y], 'w--', alpha=0.5, 
            linewidth=1, label='Aim Line')
    
    # Annotations
    max_break_idx = np.argmax(np.abs(y))
    max_break = y[max_break_idx]
    ax.annotate(f'Max Break: {abs(max_break):.1f} cm',
                xy=(x[max_break_idx], y[max_break_idx]),
                xytext=(x[max_break_idx] + 20, y[max_break_idx] + 10),
                color='white', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    ax.set_xlabel('Distance (cm)', color='white')
    ax.set_ylabel('Lateral Position (cm)', color='white')
    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', facecolor='darkgreen', 
              edgecolor='white', labelcolor='white')
    
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    if output_path:
        fig.savefig(output_path, facecolor='#228B22', edgecolor='none')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_multiple_trajectories(trajectories: List[Tuple[np.ndarray, str, str]],
                                hole_position: Tuple[float, float],
                                output_path: Optional[str] = None) -> None:
    """
    Plot multiple trajectories for comparison (e.g., different speeds).
    
    Parameters:
    -----------
    trajectories : list of (trajectory, label, color)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#228B22')
    
    hole_x = hole_position[0] * 100
    hole_y = hole_position[1] * 100
    hole_radius = 5.4
    
    for traj, label, color in trajectories:
        x = traj[:, 0] * 100
        y = traj[:, 1] * 100
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.scatter([x[-1]], [y[-1]], s=100, c=color, marker='o',
                   edgecolors='black', linewidths=1, zorder=5)
    
    # Draw hole
    hole_circle = plt.Circle((hole_x, hole_y), hole_radius,
                              color='black', fill=True, zorder=4)
    ax.add_patch(hole_circle)
    
    # Start position
    ax.scatter([0], [0], s=200, c='white', marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    
    ax.set_xlabel('Distance (cm)', color='white')
    ax.set_ylabel('Lateral Position (cm)', color='white')
    ax.set_title('Trajectory Comparison: Effect of Ball Speed', 
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', facecolor='darkgreen',
              edgecolor='white', labelcolor='white')
    
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    
    if output_path:
        fig.savefig(output_path, facecolor='#228B22', edgecolor='none')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_force_diagram(slope_percent: float,
                       output_path: Optional[str] = None) -> None:
    """
    Create a free-body diagram showing forces on a golf ball.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw inclined surface
    angle = np.arctan(slope_percent / 100)
    angle_deg = np.degrees(angle)
    
    # Surface line
    surface_length = 3
    surface_x = [0, surface_length * np.cos(angle)]
    surface_y = [0, -surface_length * np.sin(angle)]
    ax.plot(surface_x, surface_y, 'g-', linewidth=3, label='Green Surface')
    
    # Ball position
    ball_x = 1.5 * np.cos(angle)
    ball_y = -1.5 * np.sin(angle) + 0.15
    ball = plt.Circle((ball_x, ball_y), 0.15, color='white', 
                       edgecolor='black', linewidth=2)
    ax.add_patch(ball)
    
    # Force arrows
    arrow_scale = 0.5
    
    # Gravity (straight down)
    ax.annotate('', xy=(ball_x, ball_y - 0.8), xytext=(ball_x, ball_y),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(ball_x + 0.1, ball_y - 0.5, 'mg', fontsize=12, color='blue')
    
    # Normal force
    normal_x = ball_x + 0.6 * np.sin(angle)
    normal_y = ball_y + 0.6 * np.cos(angle)
    ax.annotate('', xy=(normal_x, normal_y), xytext=(ball_x, ball_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(normal_x + 0.05, normal_y, 'N', fontsize=12, color='green')
    
    # Friction force (up the slope)
    friction_x = ball_x - 0.5 * np.cos(angle)
    friction_y = ball_y + 0.5 * np.sin(angle)
    ax.annotate('', xy=(friction_x, friction_y), xytext=(ball_x, ball_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(friction_x - 0.2, friction_y + 0.1, 'f', fontsize=12, color='red')
    
    # Slope component
    slope_x = ball_x + 0.4 * np.cos(angle)
    slope_y = ball_y - 0.4 * np.sin(angle)
    ax.annotate('', xy=(slope_x, slope_y), xytext=(ball_x, ball_y),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(slope_x + 0.05, slope_y - 0.15, 'mg sin θ', fontsize=10, color='orange')
    
    # Angle arc
    arc = patches.Arc((0, 0), 0.6, 0.6, angle=0, theta1=-angle_deg, 
                       theta2=0, color='black', linewidth=1.5)
    ax.add_patch(arc)
    ax.text(0.35, -0.1, f'θ = {angle_deg:.1f}°', fontsize=10)
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.5, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Free Body Diagram: Ball on {slope_percent}% Slope',
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.plot([], [], 'b-', linewidth=2, label='Gravity (mg)')
    ax.plot([], [], 'g-', linewidth=2, label='Normal Force (N)')
    ax.plot([], [], 'r-', linewidth=2, label='Friction (f)')
    ax.plot([], [], color='orange', linewidth=2, label='Slope Component')
    ax.legend(loc='upper right')
    
    if output_path:
        fig.savefig(output_path, facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_3d_surface(slope_percent: float, 
                    size: float = 3.0,
                    output_path: Optional[str] = None) -> None:
    """
    Create a 3D visualization of a sloped putting surface.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface mesh
    x = np.linspace(0, size, 50)
    y = np.linspace(-size/2, size/2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Z based on slope (slope in Y direction)
    Z = Y * (slope_percent / 100)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='Greens', alpha=0.8,
                           linewidth=0, antialiased=True)
    
    # Draw hole
    hole_theta = np.linspace(0, 2*np.pi, 30)
    hole_radius = 0.054  # 5.4 cm
    hole_x = size * 0.8 + hole_radius * np.cos(hole_theta)
    hole_y = hole_radius * np.sin(hole_theta)
    hole_z = hole_y * (slope_percent / 100) - 0.02
    ax.plot(hole_x, hole_y, hole_z, 'k-', linewidth=2)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Height (meters)')
    ax.set_title(f'Putting Green with {slope_percent}% Side Slope',
                 fontsize=14, fontweight='bold')
    
    if output_path:
        fig.savefig(output_path, facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_heatmap(distances: np.ndarray, 
                 slopes: np.ndarray,
                 make_probs: np.ndarray,
                 output_path: Optional[str] = None) -> None:
    """
    Create a heatmap of make probability vs distance and slope.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(make_probs * 100, cmap='RdYlGn', aspect='auto',
                   origin='lower', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Make Probability (%)', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(len(distances)))
    ax.set_yticks(np.arange(len(slopes)))
    ax.set_xticklabels([f'{d:.0f}' for d in distances])
    ax.set_yticklabels([f'{s:.1f}' for s in slopes])
    
    ax.set_xlabel('Putt Distance (feet)', fontsize=12)
    ax.set_ylabel('Slope (%)', fontsize=12)
    ax.set_title('Make Probability: Distance vs Slope', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(slopes)):
        for j in range(len(distances)):
            text = ax.text(j, i, f'{make_probs[i, j]*100:.0f}',
                          ha='center', va='center', color='black',
                          fontsize=8)
    
    if output_path:
        fig.savefig(output_path, facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Visualization Module - Demo")
    print("Creating sample figures...")
    
    # Create output directory
    fig_dir = "../figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Demo force diagram
    plot_force_diagram(2.0, os.path.join(fig_dir, "force_diagram_demo.png"))
    
    # Demo 3D surface
    plot_3d_surface(2.0, output_path=os.path.join(fig_dir, "3d_surface_demo.png"))
    
    print("Demo figures created in figures/ directory")
