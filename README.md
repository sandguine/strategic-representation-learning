# strategic-representation-learning
Learning compressed representations of co-player policies based on their strategic influence using soft best responses and contrastive learning.

## Method Overview

This work is under review as a part of RLC, CoCoMARL workshop 2025. This repository implements a method for learning compressed representations of co-player policies based on their strategic influence. Instead of modeling behavior directly, the method embeds co-policies according to how they affect the ego agent's incentives i.e., how they change its best response.

### Key Components

- **Soft Best Response**: A softmax over Q-values that captures the ego agent’s probabilistic policy in response to a co-policy.
- **Strategic InfoNCE**: A contrastive loss that learns embeddings based on action-level influence.
- **ΔSEC(t)**: A metric that measures how strategic ambiguity is reduced over time during partner interaction.

---

## Learning Procedure (Strategic Embedding)

1. Sample a co-policy \( \pi_{-i} \)
2. Compute its soft best response:
   \[
   BR^\tau_i(a_i \mid \pi_{-i}) = \frac{\exp(Q_{\pi_{-i}}(a_i)/\tau)}{\sum_{a'} \exp(Q_{\pi_{-i}}(a')/\tau)}
   \]
3. Sample a positive action \( a^+ \sim BR^\tau_i(\pi_{-i}) \)
4. Sample negatives \( a^-_j \sim BR^\tau_i(\pi_j^{-}) \)
5. Use contrastive loss:
   \[
   \mathcal{L}_{\text{InfoNCE}} = -\log \frac{f(a^+, h(\pi_{-i}))}{\sum_j f(a^-_j, h(\pi_{-i}))}
   \]
6. Update encoder \( h(\pi_{-i}) \) via gradient descent

---

## Evaluation Procedure (ΔSEC Tracking)

1. Estimate co-policy \( \pi_{-i}^t \) from observations
2. Compute soft BR: \( BR^\tau_i(\pi_{-i}^t) \)
3. Identify similar co-policies in \( \mathcal{P}_{-i} \) using KL divergence:
   \[
   D_{\mathrm{KL}}(BR^\tau_i(\pi') \parallel BR^\tau_i(\pi_{-i}^t)) \leq \varepsilon
   \]
4. Define strategic neighborhood \( \mathcal{N}_\varepsilon \)
5. Compute entropy of neighborhood: \( H(S_t) \)
6. Compute ambiguity reduction:
   \[
   \Delta \mathrm{SEC}(t) = H(S_{t-1}) - H(S_t)
   \]
