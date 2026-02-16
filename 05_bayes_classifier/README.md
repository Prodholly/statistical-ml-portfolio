# Bayesian Binary Classifier

> Derivation and implementation of the MAP (Maximum A Posteriori) decision rule for a binary Gaussian generative model, computing the optimal decision boundary analytically and verifying it against numerical integration of the Bayes error rate.

---

## Problem Setup

Binary classification with the following generative model:

$$P(Y=1) = 0.3, \quad P(Y=0) = 0.7$$

$$P(X \mid Y=1) \sim \mathcal{N}(1,\,1), \quad P(X \mid Y=0) \sim \mathcal{N}(-3,\,1)$$

---

## Part (a): Posterior Probabilities

By Bayes' theorem:

$$P(Y=k \mid X=x) = \frac{p(X=x \mid Y=k)\,P(Y=k)}{p(X=x)}$$

where the marginal is $p(X=x) = \sum_k p(X \mid Y=k)\,P(Y=k)$.

| $x$ | $P(Y=1 \mid X=x)$ | $P(Y=0 \mid X=x)$ |
|-----|-------------------|-------------------|
| −4  | 0.000003          | 0.999997          |
| −2  | 0.007788          | 0.992212          |
|  0  | 0.959015          | 0.040985          |
|  1  | 0.999218          | 0.000782          |
|  3  | ≈ 1.000           | ≈ 0.000           |

---

## Part (b): MAP Decision Rule

The MAP rule predicts $\hat{y} = \arg\max_k P(Y=k \mid X=x)$, which reduces to a log-likelihood ratio threshold:

$$x \geq x^* \triangleq \frac{\log(7/3) - 4}{4} = -0.7882 \implies \hat{y} = 1$$

$$x < x^* \implies \hat{y} = 0$$

---

## Visualization

![Bayesian Classifier — Densities, Posteriors & Decision Boundary](results/bayes_classifier.png)

The left panel shows the class-conditional densities $p(X \mid Y)$ and weighted (prior-scaled) densities. The right panel shows the posterior probabilities $P(Y \mid X=x)$ crossing at $x^* = -0.788$, confirming the MAP boundary.

---

## Bayes Error Rate

The Bayes error rate is the minimum achievable error under this generative model:

$$R^* = P(\text{error} \mid Y=1)\,P(Y=1) + P(\text{error} \mid Y=0)\,P(Y=0)$$

| Component | Value |
|-----------|-------|
| $P(\text{error} \mid Y=1) \times P(Y=1)$ | 0.011062 |
| $P(\text{error} \mid Y=0) \times P(Y=0)$ | 0.009443 |
| **Bayes Error Rate** $R^*$ | **0.020505 (2.050%)** |
| **Bayes Accuracy** | **97.950%** |

The MAP rule achieves the minimum possible error for this distribution — no classifier can do better than **2.05% error** given the generative model above.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook bayes_classifier.ipynb
```

---

## Key Takeaways

- The MAP rule is equivalent to a **log-likelihood ratio test** — for Gaussians with equal variance, this reduces to a linear threshold in $x$.
- The **Bayes error rate** ($R^* \approx 2.05\%$) is the irreducible lower bound on classification error given the generative model; it arises from the overlap between the two class-conditional distributions.
- The decision boundary $x^* = -0.788$ lies closer to the class-0 mean ($\mu_0 = -3$) than to class-1 ($\mu_1 = 1$) because $P(Y=0) = 0.7 > P(Y=1) = 0.3$ — the higher prior on class 0 shifts the boundary away from class 1.
- Bayes classifiers are **generative models**: they model the full joint $P(X, Y) = p(X \mid Y)\,P(Y)$ rather than learning the boundary directly.
