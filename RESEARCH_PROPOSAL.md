# Research Proposal: The Physics of Optimal Putting Speed

## 1. Introduction and Motivation

Golf putting presents a fascinating physics problem: when putting on a sloped green, a ball curves (breaks) due to the gravitational component acting perpendicular to the intended path. A fundamental question arises: **should golfers hit putts firmly to minimize break, or softly to "die" the ball into the hole?**

This "firm vs. die" debate has practical implications but lacks rigorous physics-based analysis accessible to the general golfing community. This research aims to develop a mathematical model that quantifies the optimal putting speed as a function of slope, distance, and green speed.

## 2. Background Physics

### 2.1 Forces on a Rolling Golf Ball

A golf ball rolling on a putting green experiences:

1. **Gravitational Force:** F_g = mg (downward)
2. **Normal Force:** F_n = mg·cos(θ) (perpendicular to surface)
3. **Friction Force:** F_f = μ·F_n (opposing motion)
4. **Slope Component:** F_slope = mg·sin(θ) (along slope direction)

Where:
- m = ball mass (45.93 g)
- g = gravitational acceleration (9.81 m/s²)
- θ = slope angle
- μ = coefficient of rolling friction

### 2.2 Green Speed and Stimpmeter

The stimpmeter rating measures how far a ball rolls when released from a standard ramp. Higher ratings indicate faster (lower friction) greens:

| Rating | Description | μ (approximate) |
|--------|-------------|-----------------|
| 8-9    | Slow        | 0.065-0.075     |
| 10-11  | Medium      | 0.055-0.065     |
| 12-13  | Fast (Tour) | 0.045-0.055     |
| 14+    | Very Fast   | < 0.045         |

### 2.3 Equations of Motion

For a ball rolling on a sloped surface with initial velocity v₀ at angle α to the fall line:

**Parallel to fall line (x-direction):**
```
d²x/dt² = g·sin(θ) - μg·cos(θ)·(dx/dt)/|v|
```

**Perpendicular to fall line (y-direction):**
```
d²y/dt² = -μg·cos(θ)·(dy/dt)/|v|
```

Where |v| = √((dx/dt)² + (dy/dt)²)

## 3. Research Hypothesis

**Primary Hypothesis:** There exists an optimal initial ball velocity that maximizes make probability for putts on sloped greens, and this optimal velocity increases with slope gradient.

**Secondary Hypotheses:**
1. The relationship between optimal speed and slope is non-linear
2. Green speed significantly affects the optimal putting velocity
3. PGA Tour professionals putt closer to the theoretically optimal speed than amateur golfers

## 4. Methodology

### 4.1 Physics Model Development

1. Derive equations of motion for ball trajectory on sloped surfaces
2. Implement numerical simulation (Runge-Kutta method) in Python
3. Define "make" criteria: ball must reach hole with velocity < capture velocity
4. Calculate optimal aim point and speed for various conditions

### 4.2 Data Collection

**PGA Tour Data Sources:**
- ShotLink putting statistics (pgatour.com)
- Strokes Gained Putting data
- Make percentages by distance and slope category
- Green speed data from tournament reports

**Parameters to analyze:**
- Putt distance (5-30 feet)
- Slope gradient (0-5%)
- Green stimpmeter rating (10-14)
- Make percentage

### 4.3 Model Validation

Compare model predictions with:
1. Published physics literature on golf ball motion
2. PGA Tour make percentages by distance/slope
3. Sensitivity analysis for parameter uncertainty

## 5. Expected Results

### 5.1 Theoretical Predictions

We expect to demonstrate:
1. Break magnitude is inversely proportional to ball speed
2. Make probability follows a bell curve with respect to speed
3. Optimal speed increases approximately linearly with slope for small angles

### 5.2 Practical Applications

Results will provide:
1. Quantitative guidance for putting speed selection
2. Understanding of when "firm" vs "die" strategies are optimal
3. Insights into professional putting techniques

## 6. Originality and Contribution

This research is original because:

1. **Accessible Physics Model:** While academic papers exist on golf ball aerodynamics, few present putting physics in a form accessible to high school level
2. **Data-Driven Validation:** Combining theoretical physics with real PGA Tour statistics
3. **Practical Focus:** Addressing a real question golfers debate
4. **Novel Analysis:** Quantifying the optimal speed-slope relationship

## 7. Target Journals

### Journal of Emerging Investigators (JEI)
- Accepts high school research
- Peer-reviewed by scientists and graduate students
- Open access
- Requires novel research question

### Journal of High School Science
- Specifically for high school students
- Rigorous review process
- Values interdisciplinary work

## 8. Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Literature review, equation derivation | 2 weeks |
| 2 | Python model implementation | 2 weeks |
| 3 | Data collection and processing | 1 week |
| 4 | Simulations and analysis | 2 weeks |
| 5 | Paper writing | 3 weeks |
| 6 | Revision and peer feedback | 2 weeks |
| 7 | Journal submission | 1 week |

## 9. Required Resources

- Python programming environment
- Access to PGA Tour statistics (free, public)
- Scientific computing libraries (NumPy, SciPy, Matplotlib)
- LaTeX for paper writing
- Faculty mentor for review

## 10. References (Preliminary)

1. Penner, A.R. (2003). "The physics of golf." Reports on Progress in Physics.
2. Alessandrini, S.M. (1992). "A motivational example for the numerical solution of two-point boundary-value problems." SIAM Review.
3. PGA Tour. "ShotLink Intelligence Program." pgatour.com
4. USGA. "Stimpmeter and Green Speed." usga.org
