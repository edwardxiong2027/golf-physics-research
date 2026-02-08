"""
Enhanced Physics Model for Golf Putting Analysis

A comprehensive physics-based simulation of golf ball motion on putting
greens, modeling the relationship between initial velocity, break magnitude,
capture probability, and optimal putting speed.

Based on research by:
- Holmes (1991) - Ball-hole interaction physics
- Penner (2002, 2003) - Comprehensive putting physics
- Pelz (1977) - Optimal speed research (17-inch rule)

Author: Jordan Xiong
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pga_putting_statistics import (
    PHYSICAL_CONSTANTS, CAPTURE_VELOCITY, PELZ_OPTIMAL_SPEED,
    stimp_to_friction, get_make_percentage_df
)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

g = PHYSICAL_CONSTANTS["gravity_mps2"]
BALL_RADIUS = PHYSICAL_CONSTANTS["golf_ball_radius_m"]
BALL_MASS = PHYSICAL_CONSTANTS["golf_ball_mass_kg"]
HOLE_RADIUS = PHYSICAL_CONSTANTS["hole_radius_m"]

# Moment of inertia for solid sphere
I_BALL = (2/5) * BALL_MASS * BALL_RADIUS**2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GreenConditions:
    """
    Physical parameters describing putting green conditions.

    Attributes:
        stimpmeter: Green speed rating in feet (typically 8-14)
        slope_percent: Slope as percentage (e.g., 2.0 = 2% grade)
        slope_direction_deg: Direction of downslope relative to putt line
                            0° = downhill toward hole
                            90° = left-to-right break
                            180° = uphill away from hole
                            270° = right-to-left break
    """
    stimpmeter: float = 11.0
    slope_percent: float = 2.0
    slope_direction_deg: float = 90.0

    @property
    def slope_angle_rad(self) -> float:
        """Slope angle in radians."""
        return np.arctan(self.slope_percent / 100)

    @property
    def slope_direction_rad(self) -> float:
        """Slope direction in radians."""
        return np.radians(self.slope_direction_deg)

    @property
    def friction_coefficient(self) -> float:
        """Rolling friction coefficient derived from stimpmeter."""
        return stimp_to_friction(self.stimpmeter)

    def __str__(self) -> str:
        return (f"GreenConditions(stimp={self.stimpmeter}, "
                f"slope={self.slope_percent}%, "
                f"direction={self.slope_direction_deg}°, "
                f"μ={self.friction_coefficient:.4f})")


@dataclass
class PuttResult:
    """
    Complete results from a putting simulation.

    Attributes:
        trajectory: Array of [x, y, vx, vy] at each timestep
        times: Time values for each trajectory point
        made: Whether the putt went in
        final_position: (x, y) where ball stopped
        final_velocity: Speed when ball stopped or was captured
        max_break: Maximum lateral displacement from aim line
        distance_past_hole: How far past hole the ball would roll (negative if short)
        speed_at_hole: Ball speed when passing the hole location
        capture_type: 'center', 'edge', 'lip_out', 'miss_short', 'miss_long', 'miss_wide'
    """
    trajectory: np.ndarray
    times: np.ndarray
    made: bool
    final_position: Tuple[float, float]
    final_velocity: float
    max_break: float
    distance_past_hole: float
    speed_at_hole: Optional[float]
    capture_type: str
    aim_angle: float = 0.0
    initial_speed: float = 0.0


# =============================================================================
# EQUATIONS OF MOTION
# =============================================================================

def equations_of_motion(t: float, state: np.ndarray,
                        conditions: GreenConditions) -> np.ndarray:
    """
    Differential equations for golf ball rolling on sloped green.

    Coordinate system:
    - x-axis: Along initial aim direction (positive toward hole)
    - y-axis: Perpendicular to aim (positive to left)
    - Origin: Ball starting position

    Forces acting on ball:
    1. Gravity component along slope: F_g = m*g*sin(θ)
    2. Rolling friction: F_f = μ*m*g*cos(θ) (opposing motion)

    Args:
        t: Time (not used, but required by solver)
        state: [x, y, vx, vy] - position and velocity
        conditions: Green parameters

    Returns:
        [vx, vy, ax, ay] - velocities and accelerations
    """
    x, y, vx, vy = state

    # Current speed
    speed = np.sqrt(vx**2 + vy**2)

    # Ball has stopped
    if speed < 1e-6:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # Unit velocity vector (direction of motion)
    v_hat_x = vx / speed
    v_hat_y = vy / speed

    # Get physical parameters
    theta = conditions.slope_angle_rad      # Slope angle
    phi = conditions.slope_direction_rad     # Slope direction
    mu = conditions.friction_coefficient     # Rolling friction

    # Gravitational acceleration from slope
    # The slope causes acceleration in the downhill direction
    g_slope = g * np.sin(theta)

    # Decompose slope acceleration into x and y components
    # phi = 0° means downhill is in +x direction (toward hole)
    # phi = 90° means downhill is in +y direction (left-to-right break)
    ax_gravity = g_slope * np.cos(phi)
    ay_gravity = g_slope * np.sin(phi)

    # Rolling friction (always opposes motion)
    friction_decel = mu * g * np.cos(theta)
    ax_friction = -friction_decel * v_hat_x
    ay_friction = -friction_decel * v_hat_y

    # Total acceleration
    ax = ax_gravity + ax_friction
    ay = ay_gravity + ay_friction

    return np.array([vx, vy, ax, ay])


def ball_stopped_event(t: float, state: np.ndarray,
                       conditions: GreenConditions) -> float:
    """Event function: returns 0 when ball speed drops below threshold."""
    speed = np.sqrt(state[2]**2 + state[3]**2)
    return speed - 0.005  # Stop when speed < 5 mm/s

ball_stopped_event.terminal = True
ball_stopped_event.direction = -1


# =============================================================================
# CORE SIMULATION
# =============================================================================

def simulate_putt(distance: float, initial_speed: float, aim_angle: float,
                  conditions: GreenConditions, max_time: float = 60.0) -> PuttResult:
    """
    Simulate a single putt and determine outcome.

    Args:
        distance: Distance from ball to hole center (meters)
        initial_speed: Initial ball speed (m/s)
        aim_angle: Aim angle in radians (0 = straight at hole, positive = left)
        conditions: Green slope and speed parameters
        max_time: Maximum simulation time (seconds)

    Returns:
        PuttResult with complete trajectory and outcome analysis
    """
    # Initial conditions
    # Ball starts at origin, hole is at (distance, 0)
    x0, y0 = 0.0, 0.0
    vx0 = initial_speed * np.cos(aim_angle)
    vy0 = initial_speed * np.sin(aim_angle)

    initial_state = np.array([x0, y0, vx0, vy0])
    hole_pos = np.array([distance, 0.0])

    # Solve the differential equations
    solution = solve_ivp(
        equations_of_motion,
        t_span=(0, max_time),
        y0=initial_state,
        args=(conditions,),
        events=ball_stopped_event,
        dense_output=True,
        max_step=0.005,  # 5ms time steps for accuracy
        method='RK45'
    )

    times = solution.t
    trajectory = solution.y.T  # Shape: (N, 4)

    # Extract final state
    final_pos = trajectory[-1, :2]
    final_vel = np.sqrt(trajectory[-1, 2]**2 + trajectory[-1, 3]**2)

    # Calculate maximum break (lateral displacement)
    max_break = np.max(np.abs(trajectory[:, 1]))

    # Analyze ball-hole interaction
    made = False
    speed_at_hole = None
    capture_type = "miss_wide"
    min_distance_to_hole = float('inf')

    for i in range(len(trajectory)):
        pos = trajectory[i, :2]
        vel_vec = trajectory[i, 2:4]
        speed = np.sqrt(vel_vec[0]**2 + vel_vec[1]**2)

        dist_to_hole = np.linalg.norm(pos - hole_pos)

        if dist_to_hole < min_distance_to_hole:
            min_distance_to_hole = dist_to_hole

        # Check if ball enters the hole region
        if dist_to_hole <= HOLE_RADIUS:
            speed_at_hole = speed

            # Determine capture based on speed and entry point
            # Holmes (1991): Max capture velocity depends on entry position
            entry_offset = dist_to_hole / HOLE_RADIUS  # 0 = center, 1 = edge

            # Linear interpolation of max capture velocity
            max_capture_center = CAPTURE_VELOCITY["center_hit"]
            max_capture_edge = CAPTURE_VELOCITY["edge_hit"]
            max_capture = max_capture_center * (1 - entry_offset) + max_capture_edge * entry_offset

            if speed <= max_capture:
                made = True
                if entry_offset < 0.3:
                    capture_type = "center"
                elif entry_offset < 0.7:
                    capture_type = "half_ball"
                else:
                    capture_type = "edge"
                break
            else:
                capture_type = "lip_out"

    # Calculate distance past hole
    if made:
        distance_past_hole = 0.0
    else:
        # Project final position relative to hole
        final_x = final_pos[0]
        if final_x < distance - HOLE_RADIUS:
            distance_past_hole = final_x - distance  # Negative = short
            if capture_type == "miss_wide":
                capture_type = "miss_short"
        else:
            distance_past_hole = final_x - distance  # Positive = long
            if capture_type == "miss_wide":
                capture_type = "miss_long"

    return PuttResult(
        trajectory=trajectory,
        times=times,
        made=made,
        final_position=(final_pos[0], final_pos[1]),
        final_velocity=final_vel,
        max_break=max_break,
        distance_past_hole=distance_past_hole,
        speed_at_hole=speed_at_hole,
        capture_type=capture_type,
        aim_angle=aim_angle,
        initial_speed=initial_speed
    )


# =============================================================================
# OPTIMAL AIM FINDING
# =============================================================================

def find_optimal_aim(distance: float, initial_speed: float,
                     conditions: GreenConditions,
                     angle_range: Tuple[float, float] = (-0.5, 0.5),
                     tolerance: float = 0.001) -> Tuple[float, PuttResult]:
    """
    Find the optimal aim angle for a given speed.

    Uses optimization to find the aim that minimizes distance to hole.

    Args:
        distance: Distance to hole (meters)
        initial_speed: Ball speed (m/s)
        conditions: Green parameters
        angle_range: Range of angles to search (radians)
        tolerance: Optimization tolerance

    Returns:
        (optimal_angle, best_result)
    """
    def objective(angle: float) -> float:
        result = simulate_putt(distance, initial_speed, angle, conditions)
        # Minimize distance to hole (or 0 if made)
        if result.made:
            return -1.0  # Reward for making it
        return result.distance_past_hole**2 + (result.final_position[1])**2

    # Find optimal angle
    opt_result = minimize_scalar(objective, bounds=angle_range, method='bounded',
                                  options={'xatol': tolerance})

    best_angle = opt_result.x
    best_putt = simulate_putt(distance, initial_speed, best_angle, conditions)

    return best_angle, best_putt


# =============================================================================
# MAKE PROBABILITY CALCULATION
# =============================================================================

def calculate_make_probability(distance: float, target_speed: float,
                               conditions: GreenConditions,
                               speed_std: float = 0.06,
                               aim_std: float = 0.015,
                               n_samples: int = 500,
                               seed: int = None) -> Dict:
    """
    Monte Carlo simulation of make probability with human error.

    Models human putting error in both speed control and aim direction.

    Args:
        distance: Distance to hole (meters)
        target_speed: Intended initial speed (m/s)
        conditions: Green parameters
        speed_std: Standard deviation of speed as fraction of target (6% typical)
        aim_std: Standard deviation of aim angle (radians, ~0.015 = 0.86°)
        n_samples: Number of Monte Carlo trials
        seed: Random seed for reproducibility

    Returns:
        Dictionary with make probability and breakdown by miss type
    """
    if seed is not None:
        np.random.seed(seed)

    # First, find optimal aim for this speed
    optimal_aim, _ = find_optimal_aim(distance, target_speed, conditions)

    outcomes = {"made": 0, "lip_out": 0, "miss_short": 0, "miss_long": 0, "miss_wide": 0}

    for _ in range(n_samples):
        # Add human error
        actual_speed = target_speed * (1 + np.random.normal(0, speed_std))
        actual_aim = optimal_aim + np.random.normal(0, aim_std)

        if actual_speed > 0.1:  # Minimum reasonable speed
            result = simulate_putt(distance, actual_speed, actual_aim, conditions)
            if result.made:
                outcomes["made"] += 1
            else:
                outcomes[result.capture_type] += 1

    # Calculate probabilities
    total = sum(outcomes.values())
    probabilities = {k: v / total for k, v in outcomes.items()}

    return {
        "make_probability": probabilities["made"],
        "outcomes": outcomes,
        "probabilities": probabilities,
        "optimal_aim": optimal_aim,
        "n_samples": total
    }


# =============================================================================
# SPEED-BREAK RELATIONSHIP ANALYSIS
# =============================================================================

def analyze_speed_break_relationship(distance: float, conditions: GreenConditions,
                                      speed_range: Tuple[float, float] = (0.5, 4.0),
                                      n_speeds: int = 30) -> Dict:
    """
    Analyze how ball speed affects break magnitude.

    This is a key relationship: faster putts break less.

    Args:
        distance: Distance to hole (meters)
        conditions: Green parameters
        speed_range: Range of speeds to test (m/s)
        n_speeds: Number of speed values to test

    Returns:
        Dictionary with speed, break, and make data
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_speeds)

    results = {
        "speeds_mps": speeds,
        "speeds_mph": speeds * 2.237,  # Convert to mph
        "breaks_m": [],
        "breaks_inches": [],
        "optimal_aims_deg": [],
        "made": [],
        "distance_past_m": [],
    }

    for speed in speeds:
        opt_aim, result = find_optimal_aim(distance, speed, conditions)

        results["breaks_m"].append(result.max_break)
        results["breaks_inches"].append(result.max_break * 39.37)
        results["optimal_aims_deg"].append(np.degrees(opt_aim))
        results["made"].append(result.made)
        results["distance_past_m"].append(result.distance_past_hole)

    # Convert to numpy arrays
    for key in results:
        if key != "made":
            results[key] = np.array(results[key])

    return results


# =============================================================================
# OPTIMAL SPEED FINDING
# =============================================================================

def find_optimal_speed(distance: float, conditions: GreenConditions,
                       speed_range: Tuple[float, float] = (0.5, 4.0),
                       n_speeds: int = 25,
                       n_samples: int = 300,
                       verbose: bool = False) -> Dict:
    """
    Find the speed that maximizes make probability.

    Also returns the "17-inch rule" speed for comparison.

    Args:
        distance: Distance to hole (meters)
        conditions: Green parameters
        speed_range: Range of speeds to test (m/s)
        n_speeds: Number of speed values
        n_samples: Monte Carlo samples per speed
        verbose: Print progress

    Returns:
        Dictionary with optimal speed analysis
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_speeds)
    probabilities = []

    for i, speed in enumerate(speeds):
        if verbose:
            print(f"Testing speed {speed:.2f} m/s ({i+1}/{n_speeds})...")

        result = calculate_make_probability(
            distance, speed, conditions,
            n_samples=n_samples, seed=42+i
        )
        probabilities.append(result["make_probability"])

    probabilities = np.array(probabilities)

    # Find optimal speed
    opt_idx = np.argmax(probabilities)
    optimal_speed = speeds[opt_idx]
    max_probability = probabilities[opt_idx]

    # Find speed that results in ball stopping ~17 inches past hole
    target_past = PELZ_OPTIMAL_SPEED["optimal_distance_past"]

    pelz_speed = None
    for speed in speeds:
        _, result = find_optimal_aim(distance, speed, conditions)
        if not result.made and result.distance_past_hole > 0:
            if abs(result.distance_past_hole - target_past) < 0.05:
                pelz_speed = speed
                break

    return {
        "speeds": speeds,
        "probabilities": probabilities,
        "optimal_speed": optimal_speed,
        "max_probability": max_probability,
        "pelz_17_inch_speed": pelz_speed,
        "distance_m": distance,
        "distance_ft": distance * 3.281,
    }


# =============================================================================
# VALIDATION AGAINST PGA DATA
# =============================================================================

def validate_against_pga(conditions: GreenConditions = None,
                         distances_ft: List[float] = None) -> pd.DataFrame:
    """
    Compare model predictions to actual PGA Tour make percentages.

    Args:
        conditions: Green conditions (default: tournament standard)
        distances_ft: Distances to test (default: standard PGA distances)

    Returns:
        DataFrame comparing model vs PGA data
    """
    import pandas as pd

    if conditions is None:
        conditions = GreenConditions(stimpmeter=11.5, slope_percent=1.5,
                                      slope_direction_deg=90)

    if distances_ft is None:
        distances_ft = [3, 5, 7, 10, 15, 20, 25, 30]

    pga_data = get_make_percentage_df()
    pga_dict = dict(zip(pga_data["distance_ft"], pga_data["make_percentage"]))

    results = []

    for dist_ft in distances_ft:
        dist_m = dist_ft * 0.3048

        # Find optimal speed and calculate make probability
        opt_result = find_optimal_speed(dist_m, conditions,
                                         n_speeds=15, n_samples=200, verbose=False)

        model_make_pct = opt_result["max_probability"]
        pga_make_pct = pga_dict.get(dist_ft, np.nan)

        results.append({
            "distance_ft": dist_ft,
            "distance_m": dist_m,
            "pga_make_pct": pga_make_pct,
            "model_make_pct": model_make_pct,
            "absolute_error": abs(model_make_pct - pga_make_pct) if not np.isnan(pga_make_pct) else np.nan,
            "optimal_speed_mps": opt_result["optimal_speed"],
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED PHYSICS MODEL FOR GOLF PUTTING")
    print("=" * 70)

    # Set up tournament conditions
    conditions = GreenConditions(
        stimpmeter=12.0,
        slope_percent=2.0,
        slope_direction_deg=90  # Left-to-right break
    )

    print(f"\n{conditions}")
    print(f"Friction coefficient (μ): {conditions.friction_coefficient:.4f}")

    # Test 10-foot putt
    distance = 10 * 0.3048  # 10 feet in meters
    print(f"\nAnalyzing 10-foot putt...")

    # Speed-break analysis
    print("\n" + "-" * 50)
    print("SPEED vs BREAK ANALYSIS")
    print("-" * 50)

    sb_analysis = analyze_speed_break_relationship(distance, conditions)

    print(f"{'Speed (m/s)':<12} {'Break (in)':<12} {'Aim (°)':<12}")
    print("-" * 36)
    for i in range(0, len(sb_analysis["speeds_mps"]), 5):
        print(f"{sb_analysis['speeds_mps'][i]:<12.2f} "
              f"{sb_analysis['breaks_inches'][i]:<12.1f} "
              f"{sb_analysis['optimal_aims_deg'][i]:<12.1f}")

    # Find optimal speed
    print("\n" + "-" * 50)
    print("OPTIMAL SPEED ANALYSIS")
    print("-" * 50)

    opt_analysis = find_optimal_speed(distance, conditions, verbose=True)

    print(f"\nOptimal speed: {opt_analysis['optimal_speed']:.2f} m/s")
    print(f"Maximum make probability: {opt_analysis['max_probability']:.1%}")

    if opt_analysis['pelz_17_inch_speed']:
        print(f"Pelz '17-inch rule' speed: {opt_analysis['pelz_17_inch_speed']:.2f} m/s")

    print("\n" + "=" * 70)
    print("Demo complete!")
