# %% [markdown]
# # Strategic Representation Learning: Toy Experiments
# 
# This notebook demonstrates a minimal experiment for evaluating soft strategic equivalence classes across three types of multi-agent interactions:
# 
# - **Competitive** (e.g., Matching Pennies)
# - **Mixed-Motive** (e.g., asymmetric Coin Game)
# - **Cooperative** (e.g., Overcooked-style coordination)
# 
# We show that policies with similar effects on the ego agent's soft best response (BR) form soft strategic equivalence classes (SECs), and that SEC size and entropy vary meaningfully across settings.
# 
# ## Mathematical Framework
# 
# ### 1. Soft Best Response
# The soft best response of agent $i$ to co-policy $\pi_{-i}$ is defined as:
# 
# $$BR^\tau_i(a_i \mid \pi_{-i}) = \frac{\exp(Q_{\pi_{-i}}(a_i)/\tau)}{\sum_{a'} \exp(Q_{\pi_{-i}}(a')/\tau)}$$
# 
# where $\tau$ is the temperature parameter controlling the softness of the response.
# 
# ### 2. Strategic InfoNCE Loss
# The contrastive loss for learning strategic embeddings:
# 
# $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{f(a^+, h(\pi_{-i}))}{\sum_j f(a^-_j, h(\pi_{-i}))}$$
# 
# where $f$ is the similarity function between actions and embeddings.
# 
# ### 3. ΔSEC(t) Metric
# The reduction in strategic ambiguity over time:
# 
# $$\Delta \text{SEC}(t) = H(S_{t-1}) - H(S_t)$$
# 
# where $H(S_t)$ is the entropy of the strategic neighborhood at time $t$.

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
import ipywidgets as widgets
from IPython.display import display, Math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax

def softmax(q_values, tau=1.0):
    """Compute softmax with temperature parameter tau.
    
    Args:
        q_values: Array of Q-values
        tau: Temperature parameter (lower = harder response)
    
    Returns:
        Softmax probabilities
    """
    q = np.array(q_values)
    q = q - np.max(q)  # numerical stability
    return np.exp(q / tau) / np.sum(np.exp(q / tau))

class StrategicEmbedding(nn.Module):
    """Neural network for learning strategic embeddings."""
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embedding_dim)(x)
        return x

def create_train_state(rng, input_dim, embedding_dim, learning_rate):
    """Creates initial training state."""
    model = StrategicEmbedding(embedding_dim=embedding_dim)
    params = model.init(rng, jnp.ones((1, input_dim)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jit
def compute_infonce_loss(anchor_br, positive_br, negative_brs, tau=1.0):
    """Compute InfoNCE loss between best responses.
    
    Args:
        anchor_br: Soft best response for anchor policy
        positive_br: Soft best response for positive policy
        negative_brs: List of soft best responses for negative policies
        tau: Temperature parameter
    
    Returns:
        InfoNCE loss value
    """
    # Compute similarities using KL divergence
    pos_sim = -entropy(anchor_br, positive_br)
    neg_sims = jnp.array([-entropy(anchor_br, neg_br) for neg_br in negative_brs])
    
    # Compute InfoNCE loss
    logits = jnp.concatenate([jnp.array([pos_sim]), neg_sims]) / tau
    exp_logits = jnp.exp(logits - jnp.max(logits))
    loss = -logits[0] + jnp.log(jnp.sum(exp_logits))
    
    return loss

@jit
def train_step(state, batch):
    """Perform a single training step."""
    def loss_fn(params):
        anchor_br, positive_br, negative_brs = batch
        loss = compute_infonce_loss(anchor_br, positive_br, negative_brs)
        return loss, None

    grad_fn = grad(loss_fn, has_aux=True)
    grads, _ = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# %% [markdown]
# ### Define Policy Q-values and Strategic Influence
# 
# Each co-policy defines a vector of Q-values from the ego agent's perspective. These determine the soft best response distributions. We also define the strategic influence as the change in the ego agent's best response distribution.
# 
# The strategic influence between policies $\pi_i$ and $\pi_j$ is measured as:
# 
# $$I(\pi_i, \pi_j) = D_{KL}(BR^\tau(\pi_i) \parallel BR^\tau(\pi_j))$$
# 
# where $D_{KL}$ is the Kullback-Leibler divergence.

# %%
# Define Q-values for each setting
competitive_qs = {
    "always_head": [1, -1],
    "always_tail": [-1, 1],
    "random": [0, 0],
}

mixed_qs = {
    "greedy_red": [0.5, 0.3],
    "hovering": [0.4, 0.4],
    "blocking": [0.6, 0.2],
    "balanced": [0.45, 0.35],
}

coop_qs = {
    "clockwise": [0.8, 0.2],
    "pause_then_deliver": [0.82, 0.18],
    "idle_helper": [0.81, 0.19],
    "efficient": [0.85, 0.15],
}

games = {
    "competitive": competitive_qs,
    "mixed_motive": mixed_qs,
    "cooperative": coop_qs,
}

# %% [markdown]
# ### Interactive Temperature Parameter Exploration
# 
# Explore how the temperature parameter affects the softness of best responses and strategic equivalence classes. As $\tau \to 0$, the soft best response approaches a hard best response (one-hot distribution).

# %%
def plot_br_distributions(tau):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{setting.title()} Setting" for setting in games.keys()])
    
    for i, (setting, co_policies) in enumerate(games.items(), 1):
        for policy_name, q_values in co_policies.items():
            br = softmax(q_values, tau)
            fig.add_trace(
                go.Bar(name=policy_name, x=["Action 1", "Action 2"], y=br),
                row=1, col=i
            )
    
    fig.update_layout(height=400, width=1200, title=f"Best Response Distributions (τ={tau:.2f})")
    fig.show()

tau_slider = widgets.FloatSlider(value=0.1, min=0.01, max=2.0, step=0.01, description='Temperature (τ):')
widgets.interactive(plot_br_distributions, tau=tau_slider)

# %% [markdown]
# ### Multi-timestep ΔSEC Tracking
# 
# Track how strategic equivalence classes evolve over time as the agent interacts with different co-policies. The ΔSEC(t) metric captures the reduction in strategic ambiguity:
# 
# $$\Delta \text{SEC}(t) = H(S_{t-1}) - H(S_t)$$
# 
# where $H(S_t)$ is the entropy of the strategic neighborhood at time $t$.

# %%
def simulate_interaction_trajectory(setting, n_steps=10):
    co_policies = list(games[setting].keys())
    trajectory = []
    
    # Start with a random policy
    current_policy = np.random.choice(co_policies)
    
    for t in range(n_steps):
        # Compute current SEC
        current_br = softmax(games[setting][current_policy], tau=0.1)
        sec = []
        
        for policy in co_policies:
            br = softmax(games[setting][policy], tau=0.1)
            if entropy(current_br, br) <= 0.05:
                sec.append(policy)
        
        # Record SEC size and entropy
        sec_size = len(sec)
        sec_entropy = np.log(sec_size) if sec_size > 0 else 0
        
        trajectory.append({
            'timestep': t,
            'current_policy': current_policy,
            'sec_size': sec_size,
            'sec_entropy': sec_entropy
        })
        
        # Transition to a new policy (simulating interaction)
        current_policy = np.random.choice(co_policies)
    
    return pd.DataFrame(trajectory)

def plot_trajectory(setting):
    df = simulate_interaction_trajectory(setting)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=["SEC Size Over Time", "ΔSEC(t) Over Time"])
    
    # Plot SEC size
    fig.add_trace(
        go.Scatter(x=df['timestep'], y=df['sec_size'], mode='lines+markers', name='SEC Size'),
        row=1, col=1
    )
    
    # Plot ΔSEC(t)
    delta_sec = df['sec_entropy'].diff()
    fig.add_trace(
        go.Scatter(x=df['timestep'], y=delta_sec, mode='lines+markers', name='ΔSEC(t)'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, width=800, title=f"{setting.title()} Setting: Strategic Equivalence Evolution")
    fig.show()

setting_dropdown = widgets.Dropdown(options=list(games.keys()), description='Setting:')
widgets.interactive(plot_trajectory, setting=setting_dropdown)

# %% [markdown]
# ### Strategic Influence Visualization
# 
# Visualize how different co-policies influence the ego agent's best response distribution. The strategic influence matrix shows the KL divergence between best responses:
# 
# $$I_{ij} = D_{KL}(BR^\tau(\pi_i) \parallel BR^\tau(\pi_j))$$
# 
# where $I_{ij}$ is the influence of policy $\pi_i$ on policy $\pi_j$.

# %%
def compute_strategic_influence(base_policy, target_policy, setting):
    base_br = softmax(games[setting][base_policy], tau=0.1)
    target_br = softmax(games[setting][target_policy], tau=0.1)
    return entropy(base_br, target_br)

def plot_strategic_influence(setting):
    co_policies = list(games[setting].keys())
    influence_matrix = np.zeros((len(co_policies), len(co_policies)))
    
    for i, policy_i in enumerate(co_policies):
        for j, policy_j in enumerate(co_policies):
            influence_matrix[i, j] = compute_strategic_influence(policy_i, policy_j, setting)
    
    fig = go.Figure(data=go.Heatmap(
        z=influence_matrix,
        x=co_policies,
        y=co_policies,
        colorscale='Viridis',
        colorbar=dict(title='Strategic Influence')
    ))
    
    fig.update_layout(
        title=f"{setting.title()} Setting: Strategic Influence Matrix",
        xaxis_title="Target Policy",
        yaxis_title="Base Policy",
        height=500,
        width=600
    )
    
    fig.show()

setting_dropdown = widgets.Dropdown(options=list(games.keys()), description='Setting:')
widgets.interactive(plot_strategic_influence, setting=setting_dropdown)

# %% [markdown]
# ### Learning Progress Visualization
# 
# Track the learning progress of the strategic embeddings over time. The loss function combines InfoNCE loss with a regularization term:
# 
# $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda \|h(\pi)\|_2^2$$
# 
# where $h(\pi)$ is the embedding of policy $\pi$ and $\lambda$ is the regularization strength.

# %%
def train_embeddings(setting, n_epochs=100):
    co_policies = list(games[setting].keys())
    n_policies = len(co_policies)
    
    # Initialize model and optimizer
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_dim=2, embedding_dim=2, learning_rate=0.01)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for i in range(n_policies):
            # Get anchor, positive, and negative samples
            anchor_policy = co_policies[i]
            positive_policy = co_policies[(i + 1) % n_policies]
            negative_policies = [co_policies[j] for j in range(n_policies) if j != i and j != (i + 1) % n_policies]
            
            # Compute best responses
            anchor_br = softmax(games[setting][anchor_policy], tau=0.1)
            positive_br = softmax(games[setting][positive_policy], tau=0.1)
            negative_brs = [softmax(games[setting][p], tau=0.1) for p in negative_policies]
            
            # Create batch
            batch = (anchor_br, positive_br, negative_brs)
            
            # Update model
            state = train_step(state, batch)
            
            # Compute loss
            loss = compute_infonce_loss(anchor_br, positive_br, negative_brs)
            epoch_loss += loss
        
        losses.append(epoch_loss / n_policies)
    
    return state, losses

def plot_learning_progress(setting):
    state, losses = train_embeddings(setting)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
    
    fig.update_layout(
        title=f"{setting.title()} Setting: Learning Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        width=800
    )
    
    fig.show()

setting_dropdown = widgets.Dropdown(options=list(games.keys()), description='Setting:')
widgets.interactive(plot_learning_progress, setting=setting_dropdown) 