# %% [markdown]
# # Strategic Representation Learning: Mathematical Framework
# 
# This notebook demonstrates the core mathematical concepts of strategic equivalence classes in multi-agent interactions.
# 
# ## Mathematical Framework
# 
# ### Soft Best Response
# $$BR^\tau_i(a_i \mid \pi_{-i}) = \frac{\exp(Q_{\pi_{-i}}(a_i)/\tau)}{\sum_{a'} \exp(Q_{\pi_{-i}}(a')/\tau)}$$
# 
# ### Strategic Influence
# $$I(\pi_i, \pi_j) = D_{KL}(BR^\tau(\pi_i) \parallel BR^\tau(\pi_j))$$

# %%
import numpy as np
import pandas as pd
from scipy.stats import entropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

def softmax(q_values, tau=1.0):
    """Compute softmax with temperature parameter tau."""
    q = np.array(q_values)
    q = q - np.max(q)  # numerical stability
    return np.exp(q / tau) / np.sum(np.exp(q / tau))

# Define game settings
games = {
    "competitive": {
        "always_head": [1, -1],
        "always_tail": [-1, 1],
        "random": [0, 0],
    },
    "mixed_motive": {
        "greedy_red": [0.5, 0.3],
        "hovering": [0.4, 0.4],
        "blocking": [0.6, 0.2],
    },
    "cooperative": {
        "clockwise": [0.8, 0.2],
        "pause_then_deliver": [0.82, 0.18],
        "idle_helper": [0.81, 0.19],
    }
}

# %% [markdown]
# ### Best Response Distributions

# %%
def plot_br_distributions(tau):
    """Plot best response distributions for all settings."""
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=[f"{setting.title()} Setting" for setting in games.keys()])
    
    for i, (setting, co_policies) in enumerate(games.items(), 1):
        for policy_name, q_values in co_policies.items():
            br = softmax(q_values, tau)
            fig.add_trace(
                go.Bar(name=policy_name, x=["Action 1", "Action 2"], y=br),
                row=1, col=i
            )
    
    fig.update_layout(
        height=400, 
        width=1200, 
        title=f"Best Response Distributions (τ={tau:.2f})",
        barmode='group',
        showlegend=True
    )
    fig.show()

# Interactive temperature control
tau_slider = widgets.FloatSlider(value=0.1, min=0.01, max=2.0, step=0.01, description='Temperature (τ):')
widgets.interactive(plot_br_distributions, tau=tau_slider)

# %% [markdown]
# ### Strategic Influence Matrix

# %%
def compute_strategic_influence(base_policy, target_policy, setting, tau=0.1):
    """Compute strategic influence between policies."""
    base_br = softmax(games[setting][base_policy], tau)
    target_br = softmax(games[setting][target_policy], tau)
    return entropy(base_br, target_br)

def plot_strategic_influence(setting, tau=0.1):
    """Plot strategic influence matrix."""
    co_policies = list(games[setting].keys())
    influence_matrix = np.zeros((len(co_policies), len(co_policies)))
    
    for i, policy_i in enumerate(co_policies):
        for j, policy_j in enumerate(co_policies):
            influence_matrix[i, j] = compute_strategic_influence(policy_i, policy_j, setting, tau)
    
    fig = go.Figure(data=go.Heatmap(
        z=influence_matrix,
        x=co_policies,
        y=co_policies,
        colorscale='Viridis',
        colorbar=dict(title='Strategic Influence')
    ))
    
    fig.update_layout(
        title=f"{setting.title()} Setting: Strategic Influence Matrix (τ={tau:.2f})",
        xaxis_title="Target Policy",
        yaxis_title="Base Policy",
        height=500,
        width=600
    )
    
    fig.show()

# Interactive controls
setting_dropdown = widgets.Dropdown(options=list(games.keys()), description='Setting:')
tau_slider = widgets.FloatSlider(value=0.1, min=0.01, max=2.0, step=0.01, description='Temperature (τ):')
widgets.interactive(plot_strategic_influence, setting=setting_dropdown, tau=tau_slider)

# %% [markdown]
# ### SEC Evolution

# %%
def simulate_sec_evolution(setting, n_steps=10, tau=0.1):
    """Simulate SEC evolution over time."""
    co_policies = list(games[setting].keys())
    trajectory = []
    current_policy = np.random.choice(co_policies)
    
    for t in range(n_steps):
        current_br = softmax(games[setting][current_policy], tau)
        sec = []
        sec_distances = []
        
        for policy in co_policies:
            br = softmax(games[setting][policy], tau)
            kl_div = entropy(current_br, br)
            sec_distances.append(kl_div)
            if kl_div <= 0.05:
                sec.append(policy)
        
        trajectory.append({
            'timestep': t,
            'current_policy': current_policy,
            'sec_size': len(sec),
            'avg_distance': np.mean(sec_distances),
            'sec_members': ','.join(sec)
        })
        
        next_policy = np.random.choice(co_policies)
        trajectory[-1]['next_policy'] = next_policy
        current_policy = next_policy
    
    return pd.DataFrame(trajectory)

def plot_sec_evolution(setting, n_steps=10, tau=0.1):
    """Plot SEC evolution over time."""
    df = simulate_sec_evolution(setting, n_steps, tau)
    
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=["SEC Size Over Time", "Average KL Divergence"])
    
    # Plot SEC size
    fig.add_trace(
        go.Scatter(
            x=df['timestep'],
            y=df['sec_size'],
            mode='lines+markers',
            name='SEC Size',
            hovertemplate="Timestep: %{x}<br>" +
                         "SEC Size: %{y}<br>" +
                         "Current Policy: %{customdata[0]}<br>" +
                         "Next Policy: %{customdata[1]}<extra></extra>",
            customdata=[[row['current_policy'], row['next_policy']] for _, row in df.iterrows()]
        ),
        row=1, col=1
    )
    
    # Plot average KL divergence
    fig.add_trace(
        go.Scatter(
            x=df['timestep'],
            y=df['avg_distance'],
            mode='lines+markers',
            name='Avg KL Divergence',
            hovertemplate="Timestep: %{x}<br>" +
                         "Avg KL Div: %{y:.3f}<br>" +
                         "SEC Members: %{customdata}<extra></extra>",
            customdata=df['sec_members']
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        width=1000,
        title=f"{setting.title()} Setting: SEC Evolution (τ={tau:.2f})",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Timestep", row=1, col=1)
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Number of Policies", row=1, col=1)
    fig.update_yaxes(title_text="KL Divergence", row=2, col=1)
    
    fig.show()

# Interactive controls
setting_dropdown = widgets.Dropdown(options=list(games.keys()), description='Setting:')
tau_slider = widgets.FloatSlider(value=0.1, min=0.01, max=2.0, step=0.01, description='Temperature (τ):')
n_steps_slider = widgets.IntSlider(value=10, min=5, max=50, step=1, description='Steps:')
widgets.interactive(plot_sec_evolution, 
                   setting=setting_dropdown,
                   tau=tau_slider,
                   n_steps=n_steps_slider) 