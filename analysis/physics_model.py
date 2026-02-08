"""
Physics Model for Golf Putting on Sloped Greens

This module implements a physics-based simulation of golf ball motion
on putting greens with slope, modeling the relationship between ball
velocity, break, and make probability.

Author: Jordan Xiong
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


# Physical Constants
GOLF_BALL_MASS = 0.04593  # kg (45.93 grams)
GOLF_BALL_RADIUS = 0.02135  # m (42.7 mm diameter)
HOLE_RADIUS = 0.054  # m (4.25 inches = 108 mm diameter)
GRAVITY = 9.81  # m/s^2


@dataclass
class GreenConditions:
    """Parameters describing putting green conditions."""
    stimpmeter: float  # Stimpmeter rating (feet)
    slope_percent: float  # Slope as percentage (e.g., 2.0 = 2%)
    slope_direction: float = 0.0  # Direction of downslope in radians (0 = toward hole)
    
    @property
    def slope_angle(self) -> float:
        """Convert slope percentage to angle in radians."""
        return np.arctan(self.slope_percent / 100)
    
    @property
    def friction_coefficient(self) -> float:
        """
        Estimate rolling friction coefficient from stimpmeter.
        
        The stimpmeter measures how far a ball rolls (in feet) when 
        released from a standardized ramp. Using energy conservation:
        μ ≈ sin(20°) / (stimpmeter_in_meters)
        
        Simplified empirical relationship:
        μ ≈ 0.65 / stimpmeter
        """
        return 0.65 / self.stimpmeter


@dataclass
class PuttResult:
    """Results from a putting simulation."""
    trajectory: np.ndarray  # (N, 4) array: [x, y, vx, vy] at each timestep
    times: np.ndarray  # Time values
    final_position: Tuple[float, float]
    final_velocity: float
    total_break: float  # Lateral displacement from initial aim line
    made: bool
    distance_from_hole: float


def equations_of_motion(t: float, state: np.ndarray, 
                        conditions: GreenConditions) -> np.ndarray:
    """
    Differential equations for ball motion on sloped green.
    
    State vector: [x, y, vx, vy]
    - x: position along initial aim direction (toward hole)
    - y: position perpendicular to aim direction (break direction)
    - vx, vy: velocity components
    
    Forces:
    1. Gravity component along slope: g * sin(θ)
    2. Rolling friction: -μ * g * cos(θ) * v_hat
    
    Returns: [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    
    # Get parameters
    theta = conditions.slope_angle
    mu = conditions.friction_coefficient
    slope_dir = conditions.slope_direction
    
    # Speed (avoid division by zero)
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 1e-6:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Unit velocity vector
    vx_hat = vx / speed
    vy_hat = vy / speed
    
    # Gravitational acceleration components (slope pulls ball in slope_direction)
    g_slope = GRAVITY * np.sin(theta)
    ax_gravity = g_slope * np.cos(slope_dir)
    ay_gravity = g_slope * np.sin(slope_dir)
    
    # Friction deceleration (opposes motion)
    friction_decel = mu * GRAVITY * np.cos(theta)
    ax_friction = -friction_decel * vx_hat
    ay_friction = -friction_decel * vy_hat
    
    # Total acceleration
    ax = ax_gravity + ax_friction
    ay = ay_gravity + ay_friction
    
    return np.array([vx, vy, ax, ay])


def ball_stopped(t: float, state: np.ndarray, 
                 conditions: GreenConditions) -> float:
    """Event function: triggers when ball speed drops below threshold."""
    speed = np.sqrt(state[2]**2 + state[3]**2)
    return speed - 0.01  # Stop when speed < 0.01 m/s

ball_stopped.terminal = True
ball_stopped.direction = -1


def simulate_putt(distance: float, initial_speed: float, 
                  aim_angle: float, conditions: GreenConditions,
                  max_time: float = 30.0) -> PuttResult:
    """
    Simulate a putt and determine if it goes in.
    
    Parameters:
    -----------
    distance : float
        Distance to hole in meters
    initial_speed : float
        Initial ball speed in m/s
    aim_angle : float
        Aim angle in radians (0 = straight at hole, positive = left)
    conditions : GreenConditions
        Green slope and speed parameters
    max_time : float
        Maximum simulation time in seconds
        
    Returns:
    --------
    PuttResult with trajectory and outcome
    """
    # Initial conditions
    # Ball starts at origin, hole is at (distance, 0)
    x0, y0 = 0.0, 0.0
    vx0 = initial_speed * np.cos(aim_angle)
    vy0 = initial_speed * np.sin(aim_angle)
    
    initial_state = np.array([x0, y0, vx0, vy0])
    
    # Solve ODE
    solution = solve_ivp(
        equations_of_motion,
        t_span=(0, max_time),
        y0=initial_state,
        args=(conditions,),
        events=ball_stopped,
        dense_output=True,
        max_step=0.01
    )
    
    # Extract trajectory
    times = solution.t
    trajectory = solution.y.T  # Shape: (N, 4)
    
    final_pos = trajectory[-1, :2]
    final_vel = np.sqrt(trajectory[-1, 2]**2 + trajectory[-1, 3]**2)
    
    # Calculate break (lateral displacement)
    # Break is the y-coordinate value (perpendicular to initial aim)
    total_break = np.max(np.abs(trajectory[:, 1]))
    
    # Check if ball went in
    # Hole is at (distance, 0)
    hole_pos = np.array([distance, 0.0])
    
    # Check if trajectory passed through hole with acceptable speed
    made = False
    min_dist = float('inf')
    
    for i in range(len(trajectory)):
        pos = trajectory[i, :2]
        vel = np.sqrt(trajectory[i, 2]**2 + trajectory[i, 3]**2)
        dist_to_hole = np.linalg.norm(pos - hole_pos)
        
        if dist_to_hole < min_dist:
            min_dist = dist_to_hole
        
        # Ball is "captured" if it enters hole with reasonable speed
        # Maximum capture velocity depends on how centered the ball is
        if dist_to_hole <= HOLE_RADIUS:
            # Empirical capture velocity model
            # Center = higher capture velocity, edge = lower
            center_factor = 1 - (dist_to_hole / HOLE_RADIUS)
            max_capture_vel = 1.5 * (0.3 + 0.7 * center_factor)  # 0.45 to 1.5 m/s
            
            if vel <= max_capture_vel:
                made = True
                break
    
    return PuttResult(
        trajectory=trajectory,
        times=times,
        final_position=(final_pos[0], final_pos[1]),
        final_velocity=final_vel,
        total_break=total_break,
        made=made,
        distance_from_hole=min_dist
    )


def find_optimal_aim(distance: float, initial_speed: float,
                     conditions: GreenConditions,
                     angle_range: Tuple[float, float] = (-0.3, 0.3),
                     num_angles: int = 100) -> Tuple[float, PuttResult]:
    """
    Find the optimal aim angle for a given speed.
    
    Returns the aim angle that gets the ball closest to the hole.
    """
    angles = np.linspace(angle_range[0], angle_range[1], num_angles)
    best_angle = 0.0
    best_result = None
    min_dist = float('inf')
    
    for angle in angles:
        result = simulate_putt(distance, initial_speed, angle, conditions)
        if result.distance_from_hole < min_dist:
            min_dist = result.distance_from_hole
            best_angle = angle
            best_result = result
    
    return best_angle, best_result


def calculate_make_probability(distance: float, speed: float,
                               conditions: GreenConditions,
                               aim_uncertainty: float = 0.02,
                               speed_uncertainty: float = 0.05,
                               num_samples: int = 500) -> float:
    """
    Monte Carlo simulation of make probability with human error.
    
    Parameters:
    -----------
    aim_uncertainty : float
        Standard deviation of aim angle error (radians)
    speed_uncertainty : float
        Standard deviation of speed error (fraction of intended speed)
    num_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    Estimated make probability (0 to 1)
    """
    # First find optimal aim for this speed
    optimal_aim, _ = find_optimal_aim(distance, speed, conditions)
    
    makes = 0
    for _ in range(num_samples):
        # Add random error
        actual_aim = optimal_aim + np.random.normal(0, aim_uncertainty)
        actual_speed = speed * (1 + np.random.normal(0, speed_uncertainty))
        
        if actual_speed > 0:
            result = simulate_putt(distance, actual_speed, actual_aim, conditions)
            if result.made:
                makes += 1
    
    return makes / num_samples


def analyze_speed_vs_break(distance: float, conditions: GreenConditions,
                           speed_range: Tuple[float, float] = (0.5, 3.0),
                           num_speeds: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze how ball speed affects break magnitude.
    
    Returns arrays of speeds and corresponding break values.
    """
    speeds = np.linspace(speed_range[0], speed_range[1], num_speeds)
    breaks = []
    
    for speed in speeds:
        _, result = find_optimal_aim(distance, speed, conditions)
        breaks.append(result.total_break if result else 0)
    
    return speeds, np.array(breaks)


def find_optimal_speed(distance: float, conditions: GreenConditions,
                       speed_range: Tuple[float, float] = (0.3, 3.0),
                       num_speeds: int = 30) -> Tuple[float, float]:
    """
    Find the speed that maximizes make probability.
    
    Returns (optimal_speed, max_probability)
    """
    speeds = np.linspace(speed_range[0], speed_range[1], num_speeds)
    probabilities = []
    
    for speed in speeds:
        prob = calculate_make_probability(distance, speed, conditions,
                                         num_samples=200)
        probabilities.append(prob)
        print(f"Speed: {speed:.2f} m/s, Make Probability: {prob:.1%}")
    
    probabilities = np.array(probabilities)
    optimal_idx = np.argmax(probabilities)
    
    return speeds[optimal_idx], probabilities[optimal_idx]


def meters_to_feet(m: float) -> float:
    """Convert meters to feet."""
    return m * 3.28084


def feet_to_meters(ft: float) -> float:
    """Convert feet to meters."""
    return ft / 3.28084


# Demo and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Golf Putting Physics Model - Demo")
    print("=" * 60)
    
    # Set up conditions: 12 stimpmeter, 2% slope perpendicular to putt line
    conditions = GreenConditions(
        stimpmeter=12.0,
        slope_percent=2.0,
        slope_direction=np.pi/2  # Slope breaks left-to-right
    )
    
    print(f"\nGreen Conditions:")
    print(f"  Stimpmeter: {conditions.stimpmeter}")
    print(f"  Slope: {conditions.slope_percent}%")
    print(f"  Friction coefficient: {conditions.friction_coefficient:.4f}")
    
    # Simulate a 10-foot putt at different speeds
    distance = feet_to_meters(10)
    
    print(f"\n10-foot putt analysis:")
    print("-" * 40)
    
    speeds_mph = [1.0, 1.5, 2.0, 2.5]
    for speed in speeds_mph:
        optimal_aim, result = find_optimal_aim(distance, speed, conditions)
        print(f"Speed: {speed:.1f} m/s")
        print(f"  Optimal aim: {np.degrees(optimal_aim):.1f}°")
        print(f"  Total break: {meters_to_feet(result.total_break)*12:.1f} inches")
        print(f"  Made: {result.made}")
        print(f"  Final distance from hole: {result.distance_from_hole*100:.1f} cm")
        print()
    
    print("\nSpeed vs Break Analysis")
    print("-" * 40)
    speeds, breaks = analyze_speed_vs_break(distance, conditions)
    for s, b in zip(speeds[::4], breaks[::4]):  # Print every 4th value
        print(f"Speed: {s:.2f} m/s -> Break: {meters_to_feet(b)*12:.1f} inches")
    
    print("\n" + "=" * 60)
    print("Finding optimal putting speed (this may take a moment)...")
    print("=" * 60)
    
    optimal_speed, max_prob = find_optimal_speed(
        distance, conditions, 
        speed_range=(0.8, 2.5),
        num_speeds=15
    )
    
    print(f"\nOptimal Speed: {optimal_speed:.2f} m/s")
    print(f"Maximum Make Probability: {max_prob:.1%}")
