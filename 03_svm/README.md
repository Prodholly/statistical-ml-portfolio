# Support Vector Machine (SVM)

> From-scratch NumPy implementations of hard-margin SVM (primal & dual), kernel SVM (RBF), and soft-margin C-SVM, applied to MNIST binary classification (digit 2 vs. digit 6).

---

## Problem Statement

Learn a maximum-margin linear classifier by solving:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \; \forall i$$

The **margin** is $\frac{2}{\|\mathbf{w}\|}$ — maximizing it minimizes $\|\mathbf{w}\|^2$.

---

## Implementations

### 3.1 Hard-Margin SVM — Primal
Direct minimization of $\frac{1}{2}\|\mathbf{w}\|^2$ subject to functional margin constraints via SLSQP.

### 3.2 Hard-Margin SVM — Dual
Equivalent Lagrangian dual:
$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \mathbf{x}_i^\top\mathbf{x}_j \quad \text{s.t.} \quad \alpha_i \geq 0,\; \sum_i \alpha_i y_i = 0$$

Weight recovery: $\mathbf{w}^* = \sum_i \alpha_i^* y_i \mathbf{x}_i$. Bias recovered analytically from the geometric margin.

### 3.3 Kernel SVM — RBF Kernel
Replaces the inner product $\mathbf{x}_i^\top\mathbf{x}_j$ with a Gaussian kernel:
$$k(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$$

The kernel matrix is computed efficiently using the identity $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2\mathbf{x}_i^\top\mathbf{x}_j$, avoiding a double loop. Prediction uses: $f(\mathbf{x}) = \text{sign}\!\left(\sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b\right)$.

### 3.4 Soft-Margin SVM — C-SVM (Dual)
Introduces slack variables $\xi_i \geq 0$ allowing margin violations penalized by $C$:
$$0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

Solved for $C \in \{1, 3, 5\}$ on the full training set (11,876 samples).

---

## Dataset

| Split | Samples | Features |
|-------|---------|----------|
| Train (Problems 1–3) | 2,000 | 784 (28×28 pixels, normalized) |
| Train (Problem 4 Soft-SVM) | 11,876 | 784 |
| Test | 1,990 | 784 |

---

## Results

| Model | Train 0-1 Loss | Test 0-1 Loss | Notes |
|-------|----------------|---------------|-------|
| Hard-Margin SVM (Primal) | 0.0000 | 0.0226 | \|\|w\*\|\| = 4.26 |
| Hard-Margin SVM (Dual)   | 0.0000 | 0.0216 | 608 support vectors |
| Kernel SVM (RBF, σ=1)    | 0.0000 | 0.4814 | σ=1 too small for 784-dim; all samples become SVs |
| Soft-Margin SVM C=1      | — | — | See notebook |
| Soft-Margin SVM C=3      | — | — | See notebook |
| Soft-Margin SVM C=5      | — | — | See notebook |

**Key insight on Kernel SVM**: With σ=1 and 784-dimensional inputs, all pairwise RBF similarities collapse near zero, making all samples support vectors. In high dimensions, σ should scale as $\sigma \sim \sqrt{d}$ (here σ ≈ 28). This is a well-known curse-of-dimensionality effect on the RBF kernel.

**Primal vs. Dual agreement**: Both hard-margin formulations achieve identical training accuracy (100%) and near-identical test accuracy (~97.7–97.8%), confirming strong duality holds and the implementations are consistent.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook svm.ipynb
```

> **Note**: The soft-margin experiment (Section 6) operates on the full training set (11,876 samples) and may take 10–20 minutes due to the O(n²) SLSQP solve. Reduce `maxiter` or use a subset for faster experimentation.

---

## Key Takeaways

- The **primal** formulation is intuitive but scales poorly — the constraint count equals the number of training samples.
- The **dual** formulation is memory-efficient at inference: only support vectors (α > 0) matter.
- The **kernel trick** enables nonlinear decision boundaries without explicitly computing the feature map, but kernel hyperparameters (σ, C) are critical — especially in high dimensions.
- **Soft-margin SVM** trades margin width for misclassification tolerance via C: small C → wide margin, more violations; large C → hard-margin-like behavior.
