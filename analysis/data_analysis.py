"""
Data Analysis Module for Golf Putting Research

This module handles:
1. Loading and processing PGA Tour putting statistics
2. Statistical analysis of putting performance
3. Comparison of model predictions with real data

Author: Jordan Xiong
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# PGA Tour Putting Statistics (publicly available aggregate data)
# Source: pgatour.com/stats
PGA_MAKE_PERCENTAGES = {
    # Distance (feet): Make percentage
    3: 0.96,
    4: 0.88,
    5: 0.77,
    6: 0.66,
    7: 0.58,
    8: 0.50,
    9: 0.45,
    10: 0.40,
    15: 0.23,
    20: 0.15,
    25: 0.10,
    30: 0.07,
}

# Approximate make percentages by slope category (estimated from various sources)
# Format: {distance: {slope_category: make_percentage}}
PGA_MAKE_BY_SLOPE = {
    10: {"flat": 0.42, "slight": 0.38, "moderate": 0.32, "severe": 0.25},
    15: {"flat": 0.25, "slight": 0.22, "moderate": 0.18, "severe": 0.13},
    20: {"flat": 0.17, "slight": 0.14, "moderate": 0.11, "severe": 0.07},
}


def load_pga_data() -> pd.DataFrame:
    """
    Load PGA Tour putting statistics.
    
    In a full implementation, this would fetch from:
    - PGA Tour ShotLink data
    - Public APIs
    - Scraped statistics
    
    For now, returns curated public data.
    """
    data = []
    for distance, make_pct in PGA_MAKE_PERCENTAGES.items():
        data.append({
            "distance_ft": distance,
            "distance_m": distance * 0.3048,
            "make_percentage": make_pct,
            "source": "PGA Tour 2023 Season"
        })
    
    return pd.DataFrame(data)


def fit_make_percentage_curve(data: pd.DataFrame) -> Dict:
    """
    Fit a statistical model to putting make percentage vs distance.
    
    Uses exponential decay: P(make) = a * exp(-b * distance) + c
    
    Returns fitted parameters and goodness of fit metrics.
    """
    from scipy.optimize import curve_fit
    
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    distances = data["distance_ft"].values
    make_pcts = data["make_percentage"].values
    
    # Initial guess
    p0 = [1.0, 0.1, 0.05]
    
    # Fit curve
    popt, pcov = curve_fit(exp_decay, distances, make_pcts, p0=p0)
    
    # Calculate R-squared
    residuals = make_pcts - exp_decay(distances, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((make_pcts - np.mean(make_pcts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        "a": popt[0],
        "b": popt[1],
        "c": popt[2],
        "r_squared": r_squared,
        "std_errors": np.sqrt(np.diag(pcov)),
        "model": lambda x: exp_decay(x, *popt)
    }


def compare_model_to_pga(model_predictions: Dict[int, float],
                          pga_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compare physics model predictions to actual PGA statistics.
    
    Parameters:
    -----------
    model_predictions : dict
        {distance_ft: predicted_make_percentage}
    pga_data : DataFrame
        Actual PGA statistics
        
    Returns:
    --------
    DataFrame with comparison metrics
    """
    results = []
    
    for _, row in pga_data.iterrows():
        dist = row["distance_ft"]
        actual = row["make_percentage"]
        
        if dist in model_predictions:
            predicted = model_predictions[dist]
            error = predicted - actual
            pct_error = (error / actual) * 100 if actual > 0 else 0
            
            results.append({
                "distance_ft": dist,
                "actual_pga": actual,
                "model_predicted": predicted,
                "absolute_error": abs(error),
                "percent_error": abs(pct_error)
            })
    
    return pd.DataFrame(results)


def analyze_strokes_gained(make_pcts: Dict[int, float]) -> pd.DataFrame:
    """
    Calculate Strokes Gained baseline values for putting.
    
    Strokes Gained Putting = (Baseline putts from distance) - (Actual putts taken)
    
    The baseline is the PGA Tour average.
    """
    # PGA Tour average putts to hole out from each distance
    baseline_putts = {}
    
    for dist, make_pct in make_pcts.items():
        # Expected putts = 1 + (1 - make_pct) * expected_from_miss
        # Simplified: expected_putts ≈ 2 - make_pct (for short putts)
        # More accurate model for longer putts
        if dist <= 10:
            baseline_putts[dist] = 1 + (1 - make_pct) * 1.5
        else:
            baseline_putts[dist] = 1 + (1 - make_pct) * 1.8
    
    return pd.DataFrame([
        {"distance_ft": d, "baseline_putts": p, "make_pct": make_pcts[d]}
        for d, p in baseline_putts.items()
    ])


def statistical_tests(model_data: np.ndarray, pga_data: np.ndarray) -> Dict:
    """
    Perform statistical tests comparing model to real data.
    
    Returns:
    --------
    Dictionary with test results
    """
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(model_data, pga_data)
    
    # Correlation
    correlation, corr_pvalue = stats.pearsonr(model_data, pga_data)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((model_data - pga_data)**2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(model_data - pga_data))
    
    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "correlation": correlation,
        "correlation_pvalue": corr_pvalue,
        "rmse": rmse,
        "mae": mae
    }


def create_publication_figure(data: pd.DataFrame, 
                              model_fit: Dict,
                              output_path: str) -> None:
    """
    Create a publication-quality figure showing make percentage vs distance.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot actual data
    ax.scatter(data["distance_ft"], data["make_percentage"] * 100,
               s=100, c='navy', marker='o', label='PGA Tour Data', zorder=3)
    
    # Plot fitted curve
    x_smooth = np.linspace(3, 35, 100)
    y_smooth = model_fit["model"](x_smooth) * 100
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
            label=f'Fitted Model (R² = {model_fit["r_squared"]:.3f})')
    
    # Formatting
    ax.set_xlabel('Putt Distance (feet)', fontsize=12)
    ax.set_ylabel('Make Percentage (%)', fontsize=12)
    ax.set_title('PGA Tour Putting Make Percentage vs Distance', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def create_speed_break_figure(speeds: np.ndarray, breaks: np.ndarray,
                               output_path: str) -> None:
    """
    Create figure showing relationship between ball speed and break.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Convert to more intuitive units
    breaks_inches = breaks * 39.37  # meters to inches
    
    ax.plot(speeds, breaks_inches, 'b-', linewidth=2.5)
    ax.scatter(speeds, breaks_inches, s=60, c='darkblue', zorder=3)
    
    # Formatting
    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Total Break (inches)', fontsize=12)
    ax.set_title('Ball Speed vs Break: The Physics of "Firm" Putting', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Faster = Less Break', 
                xy=(speeds[-1], breaks_inches[-1]),
                xytext=(speeds[-1] - 0.5, breaks_inches[-1] + 2),
                fontsize=10, color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def create_optimal_speed_figure(speeds: np.ndarray, 
                                 probabilities: np.ndarray,
                                 optimal_speed: float,
                                 output_path: str) -> None:
    """
    Create figure showing make probability vs speed with optimal marked.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.plot(speeds, probabilities * 100, 'g-', linewidth=2.5)
    ax.fill_between(speeds, probabilities * 100, alpha=0.2, color='green')
    
    # Mark optimal speed
    opt_prob = probabilities[np.argmin(np.abs(speeds - optimal_speed))]
    ax.axvline(optimal_speed, color='red', linestyle='--', linewidth=2,
               label=f'Optimal Speed: {optimal_speed:.2f} m/s')
    ax.scatter([optimal_speed], [opt_prob * 100], s=150, c='red', 
               marker='*', zorder=5)
    
    # Formatting
    ax.set_xlabel('Initial Ball Speed (m/s)', fontsize=12)
    ax.set_ylabel('Make Probability (%)', fontsize=12)
    ax.set_title('Finding the Optimal Putting Speed', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("Data Analysis Module - Demo")
    print("=" * 60)
    
    # Load data
    pga_data = load_pga_data()
    print("\nPGA Tour Putting Data:")
    print(pga_data.to_string(index=False))
    
    # Fit model
    print("\n" + "-" * 40)
    print("Fitting exponential decay model...")
    fit = fit_make_percentage_curve(pga_data)
    print(f"Model: P(make) = {fit['a']:.3f} * exp(-{fit['b']:.3f} * d) + {fit['c']:.3f}")
    print(f"R-squared: {fit['r_squared']:.4f}")
    
    # Strokes gained analysis
    print("\n" + "-" * 40)
    print("Strokes Gained Baseline:")
    sg_data = analyze_strokes_gained(PGA_MAKE_PERCENTAGES)
    print(sg_data.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Demo complete. Run with actual model predictions for full analysis.")
