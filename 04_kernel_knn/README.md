# Kernel k-Nearest Neighbor Classifier

> Extension of standard k-NN that replaces Euclidean distance with a Gaussian (RBF) kernel similarity, enabling nonlinear decision boundaries and implicit high-dimensional feature space comparisons.

---

## Problem Statement

Given a query point $\mathbf{x}$, classify it by majority vote over the $k$ training points with **highest kernel similarity**:

$$\hat{y} = \text{sign}\!\left(\sum_{i \in \mathcal{N}_k^K(\mathbf{x})} y_i\right)$$

where $\mathcal{N}_k^K(\mathbf{x})$ is the set of $k$ indices maximizing $k(\mathbf{x}_i, \mathbf{x})$, with:

$$k(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$$

**Binary task**: MNIST digit **2** (label −1) vs. digit **6** (label +1).

---

## Key Distinction from Standard k-NN

| Property | Euclidean k-NN | Kernel k-NN |
|----------|---------------|-------------|
| Similarity metric | $-\|\mathbf{x}_i - \mathbf{x}_j\|_2$ | $\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma^2)$ |
| Neighbor selection | Smallest distance | Largest kernel value |
| Feature space | Input space $\mathbb{R}^d$ | Implicit RKHS $\mathcal{H}$ |
| Hyperparameter | k | k and σ |

---

## Algorithm

1. Compute the full $m \times n$ kernel matrix between test and training sets (vectorized via the identity $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2\mathbf{x}_i^\top\mathbf{x}_j$).
2. For each query point, select the $k$ training samples with highest kernel similarity using `np.argpartition` (O(n) partial sort).
3. Return $\text{sign}(\sum y_k)$ as the predicted label.

---

## Dataset

| Split | Samples | Features |
|-------|---------|----------|
| Train | 11,876 | 784 (28×28 pixels, normalized) |
| Test | 1,990 | 784 |

The full training set is used — unlike standard k-NN, vectorized kernel matrix computation keeps this tractable.

---

## Results (σ = 1.0)

| k | Train 0-1 Loss | Test 0-1 Loss | Train Acc | Test Acc |
|---|----------------|---------------|-----------|----------|
| 3 | — | — | — | — |
| 5 | — | — | — | — |

*Run the notebook to populate this table with your results.*

---

## Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Kernel matrix (m × n) | O(m · n · d) |
| k-nearest per query | O(n) via `argpartition` |
| Full inference | O(m · n · d) |
| Storage | O(n · d) — lazy learner |

Vectorized kernel computation via broadcasting eliminates all Python-level loops, giving a significant speedup over the naive O(m · n) loop implementation.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook kernel_knn.ipynb
```

> The kernel matrix for the full training set (11,876 × 11,876) requires ~530 MB of memory. Reduce to a subset if memory is limited.

---

## Key Takeaways

- The RBF kernel induces an infinite-dimensional feature space — kernel k-NN implicitly compares points in this space without ever computing the feature map explicitly.
- With σ=1 in 784 dimensions, similarities may collapse (curse of dimensionality). Tuning σ relative to the average pairwise distance is critical for meaningful comparisons.
- Kernel k-NN is strictly more expressive than Euclidean k-NN but adds a kernel hyperparameter σ that must be cross-validated.
- Unlike kernel SVM, there is no support vector compression — all training points are retained at inference time.
