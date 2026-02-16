# k-Nearest Neighbor Classifier

> From-scratch implementation of the k-NN algorithm with vectorized distance computation, evaluated on MNIST binary classification (digit 2 vs. digit 6) with a systematic sweep over k to characterize the bias-variance tradeoff.

---

## Problem Statement

Given a query point $\mathbf{x}$, classify it by a majority vote over the $k$ nearest points in the training set under Euclidean distance:

$$\hat{y} = \text{sign}\!\left(\sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i\right)$$

where $\mathcal{N}_k(\mathbf{x})$ is the set of $k$ indices minimizing $\|\mathbf{x}_i - \mathbf{x}\|_2$.

**Binary task**: MNIST digit **2** (label −1) vs. digit **6** (label +1).

---

## Algorithm

**Distance metric**: Euclidean distance in $\mathbb{R}^{784}$ pixel space.

**Prediction**:
1. Compute $\|\mathbf{x}_{\text{train},i} - \mathbf{x}_{\text{query}}\|_2$ for all training samples.
2. Select the $k$ samples with smallest distance using `np.argpartition` — O(n) vs. O(n log n) for full sort.
3. Return $\text{sign}(\sum y_k)$ as the predicted label.

**Evaluation metric**: 0-1 loss (fraction of misclassified samples).

---

## Dataset

| Split | Samples | Features |
|-------|---------|----------|
| Train (subset) | 1,000 | 784 (28×28 pixels) |
| Test | 1,990 | 784 |

A training subset of 1,000 samples is used for computational tractability — k-NN has O(n · d) prediction cost per query point, giving O(m · n · d) total for m test points.

---

## Results

| k | Train 0-1 Loss | Test 0-1 Loss | Generalization Gap |
|---|----------------|---------------|--------------------|
| 1 | 0.0000         | 0.0075        | +0.0075            |
| 3 | 0.0020         | **0.0060**    | +0.0040            |
| 5 | 0.0060         | 0.0070        | +0.0010            |
| 7 | 0.0060         | 0.0085        | +0.0025            |

**Optimal k = 3** achieves the lowest test error of **0.60%** (12/1990 misclassifications).

---

## Bias-Variance Analysis

| k | Regime | Interpretation |
|---|--------|----------------|
| k=1 | High variance | Memorizes training set (0% train error). Sensitive to noise; complex, irregular decision boundary. |
| k=3 | Balanced | Best test generalization. Smooths local noise while preserving fine-grained structure. |
| k=5,7 | Increasing bias | Decision boundary progressively smoother. Accuracy degrades as the neighborhood becomes too large to capture local class structure. |

The **generalization gap** (test − train error) peaks at k=1 and decreases through k=5, confirming the classical bias-variance tradeoff: lower k → lower bias, higher variance; higher k → higher bias, lower variance.

---

## Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Distance per query | O(n · d) |
| k-nearest selection | O(n) via `argpartition` |
| Full test set inference | O(m · n · d) |
| Storage | O(n · d) — lazy learner, no training phase |

With n=1,000, m=1,990, d=784: ≈ 1.56 billion FLOPs for the full evaluation sweep.

---

## Visualizations

- **Error vs k**: Side-by-side plot of train and test 0-1 loss across k values.
- **Generalization gap**: Test − train error as a function of k, highlighting the bias-variance tradeoff.

Results saved to `results/`.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook knn.ipynb
```

MNIST downloads automatically on first run (~11 MB). The full experiment sweep takes approximately 2–5 minutes depending on hardware.

---

## Key Takeaways

- k-NN is a **non-parametric, instance-based** learner — it stores the entire training set and defers all computation to inference time.
- The algorithm has **no explicit training phase**, but inference cost scales linearly with training set size, making it expensive at scale.
- Even with only 1,000 training samples, k-NN achieves **99.4% test accuracy** at k=3, demonstrating that digit 2 and digit 6 are highly separable in pixel space under L2 distance.
- `np.argpartition` provides a critical optimization: O(n) partial sort vs. O(n log n) full sort, with no impact on prediction quality.
