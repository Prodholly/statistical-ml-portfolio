# Linear Regression, Logistic Regression & Model Selection

> From-scratch NumPy implementations of linear regression (gradient descent), logistic regression (BGD & SGD), polynomial regression with bias-variance analysis, and k-fold cross-validation for hyperparameter selection — all applied to MNIST binary classification (digit 2 vs. digit 6).

---

## Sections

| # | Topic | Points |
|---|-------|--------|
| 1 | [Model Selection — k-SVM σ sweep, comparison table](#section-1-model-selection) | 20 |
| 2 | [Linear Regression — manual GD trace, polynomial overfitting](#section-2-linear-regression) | 20 |
| 3 | [Logistic Regression — BGD, SGD, ROC curves](#section-3-logistic-regression) | 40 |
| 4 | [K-Fold Cross Validation — 10-fold CV for σ selection](#section-4-k-fold-cross-validation) | 20 |

---

## Section 1: Model Selection

### 1(a) — k-SVM σ Sweep

Kernel SVM trained on the **first 1 000 training samples** for σ ∈ {0.2, 0.5, 1, 3, 4, 5, 10}, evaluated with 0-1 loss:

| Model | Train 0-1 Loss | Test 0-1 Loss |
|-------|:--------------:|:-------------:|
| k-SVM σ=0.2 | — | — |
| k-SVM σ=0.5 | — | — |
| k-SVM σ=1   | — | — |
| k-SVM σ=3   | — | — |
| k-SVM σ=4   | — | — |
| k-SVM σ=5   | — | — |
| k-SVM σ=10  | — | — |
| Perceptron  | — | — |
| SVM         | — | — |
| 3-NN        | — | — |
| 5-NN        | — | — |

*Run the notebook to populate — values depend on SLSQP convergence.*

![Model Selection Comparison](results/model_selection_comparison.png)

### 1(b) — Hyperparameter Tuning Discussion

- **Small σ**: kernel similarity falls off quickly → model memorises training data → overfitting (low train error, high test error).
- **Large σ**: similarity is near-constant → near-linear boundary → underfitting (both errors elevated).
- **Correct approach**: tune σ using **cross-validation on training data only** — never the test set.

### 1(d) — Model Selection

Choose the model with the lowest **cross-validation error** (not test error). The test set is used once, at the end, for unbiased evaluation. Cross-validation reveals the bias–variance tradeoff without touching the test set.

---

## Section 2: Linear Regression

### 2.1 — Manual Gradient Descent Trace

Five iterations of GD on two samples with $\theta = [0,0,0]^\top$, $\alpha = 0.1$:

$$\theta \leftarrow \theta - \alpha \sum_{i=1}^{2}(\hat{y}^{(i)} - y^{(i)})\, x^{(i)}$$

### 2.3 — Polynomial Regression: Overfitting & Underfitting

Data generated from $y = x^3 - 2x^2 + 1 + \mathcal{N}(0,1)$, fit with degrees 1–5:

![Polynomial Regression](results/polynomial_regression.png)

| Degree | Regime | Notes |
|--------|--------|-------|
| 1 | Underfitting | Linear — misses cubic curvature. High bias. |
| 2 | Underfitting | Quadratic — still too simple. |
| 3 | Good fit | Matches the true data-generating function. |
| 4 | Mild overfitting | Starts fitting noise. |
| 5 | Overfitting | Interpolates every point; near-zero train MSE. |

**Model selection**: plot train and validation MSE vs degree; choose degree where **validation MSE is minimised**.

---

## Section 3: Logistic Regression

MNIST (digit 2 vs 6), first 1 000 training samples. Labels {−1,+1} → {0,1}. Bias appended. $\theta \in \mathbb{R}^{785}$, $\alpha = 0.4$.

### Sigmoid & Loss

$$h_\theta(x) = \frac{1}{1 + e^{-\theta^\top x}}, \qquad \nabla \ell(\theta) = \sum_{i=1}^{m}(y^{(i)} - h_\theta(x^{(i)}))\, x^{(i)}$$

### 3.1 — Batch Gradient Descent

Stop when $\|\nabla \ell(\theta)\|_2 < 10^{-2}$.

![BGD Convergence](results/logistic_bgd_convergence.png)

### 3.2 — Stochastic Gradient Descent

Update on each sample; check $\|\nabla \ell\| < 10^{-3}$ every 100 steps.

### 3.3 — Dataset Usage Comparison

BGD uses the full dataset every iteration. SGD uses individual samples — far more computationally efficient per unit accuracy improvement on large datasets.

### 3.4 — ROC Curves

![ROC Curves](results/logistic_roc_curves.png)

Both BGD and SGD achieve high AUC (>0.99). BGD converges to a more precise optimum; SGD oscillates near the optimum with constant step size but achieves comparable classification quality.

---

## Section 4: K-Fold Cross Validation

10-fold CV on the **first 300 training samples** for k-SVM with σ ∈ {0.2, 0.5, 1}:

![Cross Validation](results/cross_validation.png)

The σ minimising CV error should agree with the σ minimising test error (Problem 1) — CV is the correct tool for hyperparameter selection without touching the test set.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook regression.ipynb
```

> **Note**: The k-SVM σ sweep (Section 1) trains 7 kernel SVMs on 1 000 samples each. The vectorized kernel matrix implementation is used throughout for speed, but expect ~5–15 minutes total for Section 1 + Section 4 CV.

---

## Key Takeaways

- **Linear regression** with GD converges stably for small step sizes; the update is $\theta \leftarrow \theta - \alpha X^\top(X\theta - y)$.
- **Polynomial regression** demonstrates bias–variance tradeoff directly: degree 1 underfits, degree 5 overfits, degree 3 matches the true function.
- **Logistic regression** BGD converges to a precise optimum; SGD is more efficient per unit accuracy but oscillates near convergence with constant α.
- **ROC curves** and AUC provide threshold-independent classifier comparison.
- **K-fold CV** is the principled way to select hyperparameters — it uses only training data and avoids test set contamination.
