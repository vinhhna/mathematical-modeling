# mathematical-modeling
Source code for Mathematical Modeling project: The night driving behavior in a car-following model

# Night Driving Behavior in a Car-Following Model (Reproduction)

This repository contains a complete reimplementation and reproduction of the figures from the paper:

> **Jiang, R. & Wu, Q. (2007). The night driving behavior in a car-following model. Physica A: Statistical Mechanics and its Applications.**

Each Jupyter notebook corresponds to one of the key figures in the paper and contains:
- The mathematical model setup (Full Velocity Difference model)
- Simulation logic
- Plotting code
- The reproduced figure alongside the original one for comparison

---

## 🔍 Project Structure

| File              | Description                                 |
|-------------------|---------------------------------------------|
| `fig_1.ipynb`     | Reproduces Figure 1: Optimal velocity function |
| `fig_2.ipynb`     | Reproduces Figure 2: Derivative of V(Δx)      |
| `fig_3.ipynb`     | Reproduces Figure 3: Cluster formation        |
| `fig_4.ipynb`     | Reproduces Figure 4: Fundamental diagram (small perturbation) |
| `fig_5.ipynb`     | Reproduces Figure 5: Kink–antikink waves     |
| `fig_6.ipynb`     | Reproduces Figure 6: Fundamental diagram with λ = 0.1 |
| `fig_7.ipynb`     | Reproduces Figure 7: Stable clusters         |
| `fig_8.ipynb`     | Reproduces Figure 8: Unstable clusters       |
| `fig_9.ipynb`     | Reproduces Figure 9: Randomness effects      |
| `fig_10.ipynb`    | Reproduces Figure 10: Large perturbation case |
| `fig_11.ipynb`    | Reproduces Figure 11: Single cluster         |
| `fig_12.ipynb`    | Reproduces Figure 12: Fundamental diagram, λ = 0.2 |
| `fig_13.ipynb`    | Reproduces Figure 13: Density wave inside cluster |
| `fig_14.ipynb`    | Reproduces Figure 14: Unstable clusters (λ = 0.1) |
| `helper_functions.py` | Shared simulation utilities (e.g., Euler step, initialization) |

---

## ▶️ Running the Code

### Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Jupyter Notebook

Install dependencies:
```bash
pip install numpy matplotlib notebook
