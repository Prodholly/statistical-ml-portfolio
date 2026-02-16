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
├── 03_svm/                 # Support Vector Machine (primal, dual, kernel, soft-margin)
│   ├── svm.ipynb
│   ├── results/
│   └── README.md
├── 04_kernel_knn/          # Kernel k-Nearest Neighbor (RBF kernel)
│   ├── kernel_knn.ipynb
│   ├── results/
│   └── README.md
├── 05_bayes_classifier/    # Bayesian binary classifier (MAP rule, Bayes error)
│   ├── bayes_classifier.ipynb
│   ├── results/
│   └── README.md
├── 06_regression/          # Linear regression, logistic regression, model selection, k-fold CV
│   ├── regression.ipynb
│   ├── results/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Modules

| # | Algorithm | Dataset | Key Results |
|---|-----------|---------|-------------|
| 01 | [Perceptron](./01_perceptron/) | MNIST (2 vs 6) | 97.80% test acc · 1 epoch convergence |
| 02 | [k-Nearest Neighbor](./02_knn/) | MNIST (2 vs 6) | Best k=3 · Test Loss = 0.006 |
| 03 | [Support Vector Machine](./03_svm/) | MNIST (2 vs 6) | Hard-margin, dual, RBF kernel, soft-margin C∈{1,3,5} |
| 04 | [Kernel k-NN](./04_kernel_knn/) | MNIST (2 vs 6) | RBF kernel similarity · k∈{3,5} |
| 05 | [Bayes Classifier](./05_bayes_classifier/) | Gaussian generative model | MAP rule · Bayes error = 2.05% |
| 06 | [Linear & Logistic Regression](./06_regression/) | MNIST (2 vs 6) + Polynomial | BGD/SGD · ROC · 10-fold CV |

---

## Result Previews

### 01 — Perceptron

| Convergence | Learned Weights |
|:-----------:|:---------------:|
| ![](01_perceptron/results/perceptron_convergence.png) | ![](01_perceptron/results/perceptron_weights.png) |

### 02 — k-Nearest Neighbor

![](02_knn/results/knn_error_analysis.png)

### 03 — Support Vector Machine

![](03_svm/results/svm_results_summary.png)

### 04 — Kernel k-NN

![](04_kernel_knn/results/kernel_knn_analysis.png)

### 05 — Bayes Classifier

![](05_bayes_classifier/results/bayes_classifier.png)

### 06 — Linear & Logistic Regression / Model Selection

| Model Selection | Polynomial Regression |
|:-:|:-:|
| ![](06_regression/results/model_selection_comparison.png) | ![](06_regression/results/polynomial_regression.png) |

| BGD Convergence | ROC Curves | Cross-Validation |
|:-:|:-:|:-:|
| ![](06_regression/results/logistic_bgd_convergence.png) | ![](06_regression/results/logistic_roc_curves.png) | ![](06_regression/results/cross_validation.png) |

---

## Key Technical Highlights

- **Perceptron**: Online update rule with augmented feature space (784-dim + bias), convergence in a single epoch.
- **k-NN**: Vectorized distance computation with `np.argpartition` for O(n) neighbor selection; bias-variance analysis across k.
- **SVM**: Full pipeline — primal SLSQP, Lagrangian dual, Gram matrix kernel trick, and soft-margin C-SVM with slack variables.
- **Kernel k-NN**: Vectorized RBF kernel matrix via broadcasting; implicit RKHS feature comparison without explicit feature maps.
- **Bayes Classifier**: Analytical MAP decision rule derivation for Gaussian generative model; Bayes error rate computed via numerical integration.
- **Linear & Logistic Regression**: GD trace, polynomial overfitting analysis, BGD vs SGD comparison with ROC curves and AUC, 10-fold cross-validation for σ selection.
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
