# Optimal Putting Speed: A Physics-Based Analysis of Ball Velocity, Break, and Make Probability

## Research Project for Journal Submission

**Author:** Jordan Xiong  
**Status:** In Development  
**Target Journals:** Journal of Emerging Investigators, Journal of High School Science

---

## Abstract (Draft)

This study develops a physics-based mathematical model to determine the optimal putting speed in golf, analyzing the trade-off between minimizing break (lateral deflection due to slope) and maximizing make probability. Using Newtonian mechanics, we model ball trajectory on sloped putting surfaces incorporating friction, gravity, and green speed (stimpmeter rating). The model is validated against PGA Tour ShotLink putting statistics. Our findings quantify the relationship between ball velocity, slope gradient, and make probability, providing insights into the "firm vs. die" putting debate.

---

## Project Structure

```
golf-physics-research/
├── README.md                 # Project overview
├── RESEARCH_PROPOSAL.md      # Detailed research proposal
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned and processed data
├── analysis/
│   ├── physics_model.py      # Core physics simulation
│   ├── data_analysis.py      # Statistical analysis
│   └── visualization.py      # Plotting and figures
├── figures/                  # Generated plots and diagrams
├── paper/
│   ├── drafts/              # Paper drafts
│   └── main.tex             # Final paper (LaTeX)
└── references/              # Reference materials and PDFs
```

---

## Research Question

**How does putting ball velocity affect break magnitude and make probability, and what is the optimal speed for different slope conditions?**

### Sub-questions:
1. What is the mathematical relationship between ball speed and lateral break on sloped greens?
2. How does green speed (stimpmeter rating) affect optimal putting velocity?
3. Can PGA Tour putting data validate our physics-based predictions?
4. What are the practical implications for putting strategy?

---

## Physics Concepts Applied

- **Newtonian Mechanics:** Force analysis, acceleration, velocity
- **Friction:** Rolling resistance, coefficient of friction on grass
- **Inclined Plane Motion:** Gravitational component causing break
- **Projectile Motion (2D):** Ball trajectory modeling
- **Energy Conservation:** Initial kinetic energy vs. work done by friction
- **Differential Equations:** Equations of motion for rolling ball

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: numpy, scipy, pandas, matplotlib, seaborn

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python analysis/physics_model.py
python analysis/data_analysis.py
```

---

## License

This project is for educational and research purposes.
