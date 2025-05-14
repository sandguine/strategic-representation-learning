import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit to use polling instead of inotify
st.set_option('server.fileWatcherType', 'poll')

st.set_page_config(
    page_title="Strategic Representation Learning",
    page_icon="🎮",
    layout="wide"
)

# Title and Introduction
st.title("Strategic Representation Learning")
st.markdown("""
This interactive demo explores strategic equivalence classes in multi-agent interactions.
""")

# Mathematical Framework
st.header("Mathematical Framework")
st.markdown("""
### Soft Best Response
$$BR^\\tau_i(a_i \\mid \\pi_{-i}) = \\frac{\\exp(Q_{\\pi_{-i}}(a_i)/\\tau)}{\\sum_{a'} \\exp(Q_{\\pi_{-i}}(a')/\\tau)}$$

### Strategic Influence
$$I(\\pi_i, \\pi_j) = D_{KL}(BR^\\tau(\\pi_i) \\parallel BR^\\tau(\\pi_j))$$
""")

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

# Sidebar controls
st.sidebar.title("Controls")
setting = st.sidebar.selectbox("Game Setting", list(games.keys()))
tau = st.sidebar.slider("Temperature (τ)", 0.01, 2.0, 0.1, 0.01)
n_steps = st.sidebar.slider("Number of Steps", 5, 50, 10, 1)

# Best Response Distributions
st.header("Best Response Distributions")
fig_br = make_subplots(rows=1, cols=3, 
                      subplot_titles=[f"{s.title()} Setting" for s in games.keys()])

for i, (s, co_policies) in enumerate(games.items(), 1):
    for policy_name, q_values in co_policies.items():
        br = softmax(q_values, tau)
        fig_br.add_trace(
            go.Bar(name=policy_name, x=["Action 1", "Action 2"], y=br),
            row=1, col=i
        )

fig_br.update_layout(
    height=400, 
    width=1200, 
    title=f"Best Response Distributions (τ={tau:.2f})",
    barmode='group',
    showlegend=True
)
st.plotly_chart(fig_br, use_container_width=True)

# Strategic Influence Matrix
st.header("Strategic Influence Matrix")

def compute_strategic_influence(base_policy, target_policy, setting, tau=0.1):
    """Compute strategic influence between policies."""
    base_br = softmax(games[setting][base_policy], tau)
    target_br = softmax(games[setting][target_policy], tau)
    return entropy(base_br, target_br)

co_policies = list(games[setting].keys())
influence_matrix = np.zeros((len(co_policies), len(co_policies)))

for i, policy_i in enumerate(co_policies):
    for j, policy_j in enumerate(co_policies):
        influence_matrix[i, j] = compute_strategic_influence(policy_i, policy_j, setting, tau)

fig_influence = go.Figure(data=go.Heatmap(
    z=influence_matrix,
    x=co_policies,
    y=co_policies,
    colorscale='Viridis',
    colorbar=dict(title='Strategic Influence')
))

fig_influence.update_layout(
    title=f"{setting.title()} Setting: Strategic Influence Matrix (τ={tau:.2f})",
    xaxis_title="Target Policy",
    yaxis_title="Base Policy",
    height=500,
    width=600
)
st.plotly_chart(fig_influence, use_container_width=True)

# SEC Evolution
st.header("SEC Evolution")

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

df = simulate_sec_evolution(setting, n_steps, tau)

fig_sec = make_subplots(rows=2, cols=1,
                       subplot_titles=["SEC Size Over Time", "Average KL Divergence"])

# Plot SEC size
fig_sec.add_trace(
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
fig_sec.add_trace(
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

fig_sec.update_layout(
    height=800,
    width=1000,
    title=f"{setting.title()} Setting: SEC Evolution (τ={tau:.2f})",
    showlegend=True
)

fig_sec.update_xaxes(title_text="Timestep", row=1, col=1)
fig_sec.update_xaxes(title_text="Timestep", row=2, col=1)
fig_sec.update_yaxes(title_text="Number of Policies", row=1, col=1)
fig_sec.update_yaxes(title_text="KL Divergence", row=2, col=1)

st.plotly_chart(fig_sec, use_container_width=True)

# Add explanation section
st.header("Understanding the Visualizations")
st.markdown("""
### Best Response Distributions
- Shows how different policies respond to each other
- Higher temperature (τ) leads to more uniform distributions
- Lower temperature makes the distributions more peaked

### Strategic Influence Matrix
- Measures how much one policy influences another
- Darker colors indicate stronger influence
- Diagonal shows self-influence

### SEC Evolution
- Tracks how strategic equivalence classes change over time
- SEC Size: Number of policies that are strategically equivalent
- KL Divergence: Measure of how different the policies are
""") 