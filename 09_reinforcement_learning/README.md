# Reinforcement Learning — GridWorld

> From-scratch implementations of TD(0), Q-Learning, and SARSA on a 4×4 GridWorld environment, with comparison of V(S1) estimates across algorithms and ε values.

---

## Environment

A 4×4 grid with states S1–S16. The agent starts at S1 (top-left) and the terminal state is S16 (bottom-right, reward +20). All other states have fixed immediate rewards ranging from −10 to +3.

| S1 (r=0) | S2 (r=−1) | S3 (r=−5) | S4 (r=+3) |
|:---------:|:---------:|:---------:|:---------:|
| S5 (r=−3) | S6 (r=−2) | S7 (r=−6) | S8 (r=−2) |
| S9 (r=−2) | S10 (r=−4) | S11 (r=−8) | S12 (r=−10) |
| S13 (r=+2) | S14 (r=−3) | S15 (r=−9) | **S16 (r=+20, terminal)** |

**Actions**: up, down, left, right (boundary moves are absorbed — agent stays in place).

**Hyperparameters**: γ = 0.9, α = 0.1, 500,000 iterations per run.

---

## Task 1 — TD(0) Value Estimation

I ran TD(0) with a uniform random policy (each available action equally likely) to estimate the state-value function $V^\pi(s)$ for all states.

$$V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right]$$

### Value Function Heatmap

![TD Value Function](results/td_value_function.png)

The heatmap shows that states near S16 have the highest values, with values decreasing as distance increases. States along high-penalty paths (S11=−8, S12=−10, S15=−9) have the lowest values. V(S1) ≈ −22.5 under the random policy, reflecting the expected negative accumulated reward before reaching S16 by chance.

---

## Task 2 — Q-Learning

Q-Learning is an off-policy algorithm that learns the optimal action-value function by always bootstrapping from the maximum Q-value of the next state:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right]$$

### Learned Greedy Policy

![Q-Learning Policy](results/qlearning_policy.png)

The greedy policy learned by Q-Learning routes the agent through S4 (r=+3) and S8, avoiding the high-penalty column (S11, S12, S15). V(S1) under this greedy policy is 0.7208 — far better than the random-policy TD estimate, since Q-Learning optimizes for the best reachable outcome.

---

## Task 3 — SARSA (ε-greedy)

SARSA is an on-policy algorithm: it updates Q using the action actually selected under the current ε-greedy policy, not the greedy maximum.

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma Q(s', a') - Q(s,a)\right]$$

where $a'$ is chosen by the ε-greedy policy. I ran SARSA for ε ∈ {0.05, 0.1, 0.2, 0.5}.

| ε | V(S1) |
|---|-------|
| 0.05 | 0.7208 |
| 0.10 | 0.5379 |
| 0.20 | 0.7208 |
| 0.50 | −5.2629 |

---

## Task 4 — Algorithm Comparison

![RL Comparison](results/rl_comparison.png)

Q-Learning and SARSA with ε=0.05 and ε=0.2 converged to V(S1) ≈ 0.72. SARSA with ε=0.1 reached 0.54 due to stochastic update variance. SARSA with ε=0.5 performed worst (V(S1) ≈ −5.26) — at 50% random exploration, the on-policy updates were so heavily corrupted by random actions that the agent never reliably learned the path to S16.

The key structural difference: Q-Learning decouples the exploration policy (random) from the target policy (greedy), so the learned Q always reflects the optimal trajectory. SARSA's on-policy update means that at high ε, it is effectively learning the value of a highly random policy, not the greedy one.

---

## How to Run

```bash
pip install -r ../requirements.txt
jupyter notebook reinforcement_learning.ipynb
```

> Each algorithm runs 500,000 iterations — expect ~1–2 minutes per run (5 total runs: 1 TD, 1 Q-Learning, 4 SARSA).

---

## Key Takeaways

- TD(0) with a random policy gave V(S1) ≈ −22.5 — the agent rarely reaches S16 by chance and accumulates large negative rewards along the way.
- Q-Learning (off-policy) converged to V(S1) = 0.72 under its greedy policy — the learned policy avoids all high-penalty states and routes through the +3 reward state.
- SARSA (on-policy) matched Q-Learning at low ε but degraded sharply at ε=0.5, because on-policy updates propagate the cost of random exploration into the Q estimates.
- The on-policy vs off-policy distinction is the fundamental difference between SARSA and Q-Learning — it only matters in practice when exploration is significant.
