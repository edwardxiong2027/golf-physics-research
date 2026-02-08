#!/usr/bin/env python3
"""
Main Runner Script for Golf Putting Physics Research

This script runs all analyses and generates all outputs for the research paper.

Usage:
    python run_analysis.py [--quick] [--figures-only] [--validation-only]

Author: Jordan Xiong
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_physics_model_demo():
    """Run the enhanced physics model demonstration."""
    print_header("RUNNING PHYSICS MODEL DEMONSTRATION")

    from analysis.enhanced_physics_model import (
        GreenConditions, simulate_putt, find_optimal_aim,
        analyze_speed_break_relationship, find_optimal_speed
    )

    # Set up tournament conditions
    conditions = GreenConditions(
        stimpmeter=12.0,
        slope_percent=2.0,
        slope_direction_deg=90
    )

    print(f"Green Conditions: {conditions}")
    print(f"Friction coefficient (μ): {conditions.friction_coefficient:.4f}")

    # Test 10-foot putt
    distance_ft = 10
    distance_m = distance_ft * 0.3048
    print(f"\nAnalyzing {distance_ft}-foot putt...")

    # Speed-break analysis
    print("\n--- Speed vs Break Analysis ---")
    sb = analyze_speed_break_relationship(distance_m, conditions, n_speeds=15)

    print(f"{'Speed (m/s)':<12} {'Break (in)':<12} {'Aim (°)':<10}")
    print("-" * 34)
    for i in [0, 4, 7, 10, 14]:
        if i < len(sb["speeds_mps"]):
            print(f"{sb['speeds_mps'][i]:<12.2f} "
                  f"{sb['breaks_inches'][i]:<12.1f} "
                  f"{sb['optimal_aims_deg'][i]:<10.1f}")

    # Optimal speed
    print("\n--- Finding Optimal Speed ---")
    opt = find_optimal_speed(distance_m, conditions,
                              n_speeds=12, n_samples=200, verbose=True)

    print(f"\n✓ Optimal speed: {opt['optimal_speed']:.2f} m/s")
    print(f"✓ Maximum make probability: {opt['max_probability']:.1%}")

    return True


def run_data_analysis():
    """Run data analysis module."""
    print_header("RUNNING DATA ANALYSIS")

    from data.pga_putting_statistics import (
        get_make_percentage_df, get_slope_effect_df, strokes_gained_baseline
    )

    # Load PGA data
    pga_df = get_make_percentage_df()
    print("PGA Tour Make Percentage by Distance:")
    print(pga_df.to_string(index=False))

    # Slope effects
    print("\n--- Slope Effects on Make Rate ---")
    slope_df = get_slope_effect_df()
    print(slope_df.head(12).to_string(index=False))

    # Strokes gained baseline
    print("\n--- Strokes Gained Baseline ---")
    for d in [5, 10, 15, 20, 30]:
        print(f"  {d} ft: {strokes_gained_baseline(d):.2f} expected putts")

    return True


def generate_figures():
    """Generate all publication figures."""
    print_header("GENERATING PUBLICATION FIGURES")

    try:
        from analysis.generate_figures import generate_all_figures
        generate_all_figures()
        return True
    except Exception as e:
        print(f"Error generating figures: {e}")
        print("Note: Some figures require matplotlib display. Run locally if on server.")
        return False


def run_validation():
    """Run model validation against PGA data."""
    print_header("MODEL VALIDATION")

    from analysis.enhanced_physics_model import GreenConditions, find_optimal_speed
    from data.pga_putting_statistics import get_make_percentage_df

    conditions = GreenConditions(stimpmeter=11.5, slope_percent=1.5,
                                  slope_direction_deg=90)

    pga_data = get_make_percentage_df()
    distances_ft = [5, 10, 15, 20]

    print("Comparing model predictions to PGA Tour data...")
    print(f"\n{'Distance':<10} {'PGA Tour':<12} {'Model':<12} {'Error':<10}")
    print("-" * 44)

    model_preds = []
    pga_vals = []

    for dist_ft in distances_ft:
        dist_m = dist_ft * 0.3048

        # Get PGA value
        pga_row = pga_data[pga_data["distance_ft"] == dist_ft]
        if len(pga_row) > 0:
            pga_pct = pga_row["make_percentage"].values[0]
        else:
            continue

        # Get model prediction
        print(f"  Computing {dist_ft}ft...", end=" ", flush=True)
        result = find_optimal_speed(dist_m, conditions,
                                     n_speeds=10, n_samples=150, verbose=False)
        model_pct = result["max_probability"]
        print("done")

        error = abs(model_pct - pga_pct)

        print(f"{dist_ft} ft{'':<5} {pga_pct*100:<12.1f} {model_pct*100:<12.1f} {error*100:<10.1f}")

        model_preds.append(model_pct)
        pga_vals.append(pga_pct)

    # Calculate correlation
    import numpy as np
    if len(model_preds) >= 3:
        corr = np.corrcoef(model_preds, pga_vals)[0, 1]
        print(f"\nCorrelation: r = {corr:.3f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run golf putting physics analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick demo only")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures")
    parser.add_argument("--validation-only", action="store_true",
                        help="Only run validation")

    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 70)
    print("   GOLF PUTTING PHYSICS RESEARCH - ANALYSIS RUNNER")
    print("=" * 70)

    if args.figures_only:
        generate_figures()
    elif args.validation_only:
        run_validation()
    elif args.quick:
        run_physics_model_demo()
    else:
        # Run full analysis
        run_physics_model_demo()
        run_data_analysis()
        run_validation()
        generate_figures()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"   COMPLETED in {elapsed:.1f} seconds")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
