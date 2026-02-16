# Statistical Machine Learning — Implementation Portfolio

> End-to-end implementations of core statistical machine learning algorithms from scratch using NumPy, evaluated on benchmark image classification tasks.

---

## Overview

This repository contains rigorous implementations of fundamental supervised learning algorithms, developed as part of a graduate-level Statistical Machine Learning course. Each module covers theoretical foundations, a clean from-scratch implementation, empirical evaluation, and analysis of learning dynamics including convergence behavior and bias-variance tradeoffs.

All experiments are conducted on the **MNIST handwritten digit dataset** (binary classification: digit 2 vs. digit 6), providing a consistent benchmark across algorithms.

---

## Repository Structure

```
statistical-ml-portfolio/
├── 01_perceptron/          # Perceptron learning algorithm
│   ├── perceptron.ipynb
│   ├── results/
│   └── README.md
├── 02_knn/                 # k-Nearest Neighbor classifier
│   ├── knn.ipynb
│   ├── results/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Modules

| # | Algorithm | Dataset | Key Results |
|---|-----------|---------|-------------|
| 01 | [Perceptron](./01_perceptron/) | MNIST (2 vs 6) | 97.8% train acc · 97.79% test acc · 1 epoch |
| 02 | [k-Nearest Neighbor](./02_knn/) | MNIST (2 vs 6) | Best k=3 · Test Loss = 0.006 |

---

## Key Technical Highlights

- **Perceptron**: Online update rule with augmented feature space (784-dim pixel features + bias), convergence in a single epoch on a linearly separable binary subset.
- **k-NN**: Vectorized distance computation with `np.argpartition` for O(n) neighbor selection; systematic sweep over k to expose the bias-variance tradeoff.
- All implementations are written in **pure NumPy** — no scikit-learn model APIs — demonstrating deep understanding of the underlying mathematics.

---

## Setup

```bash
git clone https://github.com/Prodholly/statistical-ml-portfolio.git
cd statistical-ml-portfolio
pip install -r requirements.txt
```

Then open any module notebook:

```bash
jupyter notebook 01_perceptron/perceptron.ipynb
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `matplotlib` | Visualization |
| `tensorflow` | MNIST dataset loader only |
| `jupyter` | Notebook environment |

---

## Author

**Emmanuel** · [@Prodholly](https://github.com/Prodholly)
