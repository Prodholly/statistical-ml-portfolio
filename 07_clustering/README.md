# K-Means Clustering

> From-scratch vectorized implementation of the K-Means algorithm applied to a 2D dataset, with sensitivity analysis across k ∈ {2, 3, 4} and four different random initializations per k.

---

## Problem Setup

Given an unlabeled 2D dataset of 600 points, I ran K-Means clustering to partition the data into k groups by minimizing within-cluster sum of squares (WCSS / inertia). The core question was how the choice of k and the random initialization affect the final cluster assignments.

---

## Algorithm

I implemented K-Means from scratch using vectorized NumPy distance computation:

$$d_{ij}^2 = \|x_i - c_j\|^2 = \|x_i\|^2 + \|c_j\|^2 - 2\,x_i^\top c_j$$

This avoids explicit loops over data points, computing the full distance matrix in a single matrix operation. The algorithm proceeds as:

1. Initialize k centroids by sampling k random training points (without replacement).
2. Assign each point to its nearest centroid: $z_i = \arg\min_j d_{ij}^2$.
3. Update centroids as the mean of each assigned cluster.
4. Repeat until centroid positions converge (`np.allclose`).

---

## Dataset

| Property | Value |
|----------|-------|
| Samples | 600 |
| Features | 2 |
| Source | `hw4_data.mat` |

---

## Results — k = 2, 3, 4

### k = 2

![K-Means k=2](results/kmeans_k2.png)

With k=2, the algorithm merges two of the three natural clusters into one group. The boundary varies slightly across initializations depending on which cluster gets split, but all four runs converge to a stable two-cluster partition.

### k = 3

![K-Means k=3](results/kmeans_k3.png)

k=3 matched the natural structure of the data. Most initializations converged to the same solution; one or two with unlucky starting points landed in a local minimum with slightly higher inertia.

### k = 4

![K-Means k=4](results/kmeans_k4.png)

With k=4, the algorithm split one of the natural clusters into two. The assignment of which cluster gets split varied with initialization, leading to higher variance in inertia across runs.

---

## Elbow Method

![Elbow Curve](results/kmeans_elbow.png)

I computed the best inertia (across 4 seeds) for k = 1 to 8. The elbow appears at k=3 — inertia drops steeply from k=1 to k=3 and then flattens, confirming that three clusters best describes the data structure without overfitting.

---

## Initialization Sensitivity

K-Means is sensitive to initialization: different random seeds can produce different cluster assignments, especially when clusters overlap or when k is misspecified. The inertia values reported in each subplot panel show that k=3 had the lowest variance across initializations, while k=4 showed the most instability.

---

## How to Run

```bash
pip install -r ../requirements.txt
# place hw4_data.mat in this directory
jupyter notebook clustering.ipynb
```

---

## Key Takeaways

- Vectorized distance computation via the squared-norm expansion eliminates inner Python loops and scales efficiently to large datasets.
- k=3 recovered the true data structure — confirmed by the elbow curve and lowest cross-initialization variance.
- K-Means is sensitive to initialization: running multiple seeds and selecting the lowest-inertia solution is standard practice.
- Over-specifying k (k=4) caused one natural cluster to be split, increasing inertia variance across initializations.
