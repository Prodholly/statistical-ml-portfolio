# Neural Network — Fashion MNIST Classification

> Feedforward neural network built with TensorFlow/Keras to classify the Fashion MNIST dataset into 10 clothing categories, with training history analysis and per-class prediction visualization.

---

## Problem Setup

I built and trained a multi-layer feedforward network on the Fashion MNIST dataset — 60,000 training and 10,000 test grayscale images (28×28) across 10 clothing categories. The goal was to train a classifier achieving strong test accuracy while observing how training and validation loss/accuracy evolve over epochs.

---

## Dataset

| Split | Samples | Shape |
|-------|---------|-------|
| Train | 60,000 | 28×28 |
| Test  | 10,000 | 28×28 |

**Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Pixel values normalized to [0, 1] by dividing by 255.

### Sample Images

![Fashion MNIST Sample Grid](results/fashion_sample_grid.png)

---

## Architecture

$$\text{Flatten}(784) \to \text{Dense}(128, \text{ReLU}) \to \text{Dense}(64, \text{ReLU}) \to \text{Dense}(10, \text{Softmax})$$

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Flatten | (784,) | 0 |
| Dense + ReLU | (128,) | 100,480 |
| Dense + ReLU | (64,) | 8,256 |
| Dense + Softmax | (10,) | 650 |
| **Total** | | **109,386** |

**Optimizer**: Adam · **Loss**: Sparse categorical cross-entropy · **Epochs**: 5

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **86.88%** |
| Test Loss | **0.3790** |

### Training History

![Training History](results/nn_training_history.png)

Training accuracy rose from ~82% in epoch 1 to ~89% by epoch 5. Validation accuracy tracked closely with no sign of overfitting over 5 epochs — both train and validation loss decreased monotonically, indicating stable convergence under Adam.

### Predictions on Test Samples

![Test Predictions](results/nn_predictions.png)

Most test samples were classified correctly (green). Misclassifications clustered on visually similar pairs — Shirt vs T-shirt/top, and Coat vs Pullover — which share similar pixel distributions and are genuinely hard to distinguish even for humans. The 86–88% test accuracy reflects this inter-class overlap.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook neural_network.ipynb
```

> Fashion MNIST is downloaded automatically via `tf.keras.datasets.fashion_mnist.load_data()`.

---

## Key Takeaways

- A two-hidden-layer feedforward network (128→64→10) achieved ~87% test accuracy on Fashion MNIST in just 5 epochs with Adam.
- Train and validation curves tracked closely — no overfitting detected, suggesting the model could benefit from more epochs or a larger architecture before regularization is needed.
- The hardest categories to classify were visually similar pairs (Shirt/T-shirt, Coat/Pullover), which would require deeper features or data augmentation to disambiguate.
- Softmax output gives a probability distribution over all 10 classes, making the model well-calibrated for threshold-based decisions.
