"""
Streamlit dashboard for Hybrid Parallel BFS Benchmark Results.

Reads ./results/benchmark.csv and renders four Plotly charts:
  1. Speedup curve    — x=num_ranks, y=speedup
  2. TEPS bar chart   — grouped by (ranks × threads) configuration
  3. Compute vs Comm  — stacked bar per configuration
  4. Graph preview    — NetworkX sample of ≤500 vertices drawn as Plotly scatter

Also displays a hardware configuration metric card at the top.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import os
import glob

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Parallel BFS Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a dark, premium feel ──────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
  }
  .metric-sub {
    font-size: 0.7rem;
    color: #484f58;
    margin-top: 4px;
  }
  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9d1d9;
    border-left: 3px solid #58a6ff;
    padding-left: 10px;
    margin: 24px 0 12px;
  }
  .stPlotlyChart { border-radius: 10px; }
  div[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#30363d", zeroline=False),
    yaxis=dict(gridcolor="#30363d", zeroline=False),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLOR_SEQ = px.colors.qualitative.Bold

def fmt_metric(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ BFS Dashboard")
st.sidebar.markdown("---")

CSV_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "benchmark.csv"
)
csv_path = st.sidebar.text_input("CSV path", value=CSV_PATH_DEFAULT)
graph_sample_path = st.sidebar.text_input(
    "Graph edge-list (optional preview)",
    value=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "roadNet-CA.txt"
    )
)
sample_size = st.sidebar.slider("Graph preview vertices", 50, 500, 250, step=50)
st.sidebar.markdown("---")
st.sidebar.caption("Hybrid Parallel BFS · MPI + Threads · Julia")

# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────

st.title("⚡ Hybrid Parallel BFS — Benchmark Dashboard")

@st.cache_data(ttl=30)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    
    # Drop duplicate header rows (if MPI concurrent writes caused duplicates)
    df = df[df["ranks"] != "ranks"].copy()
    
    # Convert expected numeric columns safely
    numeric_cols = [
        "ranks", "threads_per_rank", "num_vertices", "num_edges", 
        "mean_time_s", "stddev_time_s", "teps", "mean_compute_s", 
        "mean_comm_s", "serial_time_s", "speedup"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df["config"] = df.apply(
        lambda r: f"{int(r['ranks'])}R × {int(r['threads_per_rank'])}T", axis=1
    )
    df["total_cores"] = df["ranks"] * df["threads_per_rank"]
    df = df.sort_values(["ranks", "threads_per_rank"])
    return df

df = load_csv(csv_path)

if df.empty:
    st.warning(
        f"No benchmark data found at `{csv_path}`. "
        "Run the Julia MPI benchmark first, then refresh this page."
    )
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# Metric cards row
# ────────────────────────────────────────────────────────────────────────────

latest = df.iloc[-1]
best_speedup_row = df.loc[df["speedup"].idxmax()]
best_teps_row    = df.loc[df["teps"].idxmax()]

st.markdown('<div class="section-header">Hardware Configuration Summary</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(fmt_metric("Ranks", str(int(latest["ranks"]))), unsafe_allow_html=True)
c2.markdown(fmt_metric("Threads / Rank", str(int(latest["threads_per_rank"]))),
            unsafe_allow_html=True)
c3.markdown(fmt_metric("Total Cores",
                        str(int(latest["ranks"] * latest["threads_per_rank"]))),
            unsafe_allow_html=True)
c4.markdown(fmt_metric("Best Speedup",
                        f"{best_speedup_row['speedup']:.2f}×",
                        best_speedup_row["config"]),
            unsafe_allow_html=True)
c5.markdown(fmt_metric("Peak TEPS",
                        f"{best_teps_row['teps']/1e6:.1f}M",
                        best_teps_row["config"]),
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Chart 1 — Speedup curve
# ────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Chart 1 — Speedup vs. Number of Ranks</div>',
            unsafe_allow_html=True)

thread_groups = df["threads_per_rank"].unique()
fig1 = go.Figure()

for i, t in enumerate(sorted(thread_groups)):
    subset = df[df["threads_per_rank"] == t].sort_values("ranks")
    fig1.add_trace(go.Scatter(
        x=subset["ranks"],
        y=subset["speedup"],
        mode="lines+markers",
        name=f"{int(t)} thread(s)/rank",
        line=dict(width=2.5, color=COLOR_SEQ[i % len(COLOR_SEQ)]),
        marker=dict(size=8, symbol="circle"),
    ))

# Ideal linear speedup reference
max_ranks = int(df["ranks"].max())
fig1.add_trace(go.Scatter(
    x=list(range(1, max_ranks + 1)),
    y=list(range(1, max_ranks + 1)),
    mode="lines",
    name="Ideal (linear)",
    line=dict(dash="dash", color="#6e7681", width=1.5),
))

fig1.update_layout(
    **PLOTLY_LAYOUT,
    title="Parallel Speedup Curve",
    xaxis_title="Number of MPI Ranks",
    yaxis_title="Speedup (serial / parallel)",
    height=420,
)
st.plotly_chart(fig1, width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# Chart 2 — TEPS bar chart
# ────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Chart 2 — TEPS by Configuration</div>',
            unsafe_allow_html=True)

fig2 = px.bar(
    df.sort_values("teps", ascending=False),
    x="config",
    y="teps",
    color="config",
    color_discrete_sequence=COLOR_SEQ,
    text=df.sort_values("teps", ascending=False)["teps"]
          .apply(lambda x: f"{x/1e6:.2f}M"),
    labels={"teps": "TEPS", "config": "Configuration"},
)
fig2.update_traces(textposition="outside", cliponaxis=False)
fig2.update_layout(
    **PLOTLY_LAYOUT,
    title="Traversed Edges Per Second (TEPS)",
    xaxis_title="Configuration (Ranks × Threads)",
    yaxis_title="TEPS",
    showlegend=False,
    height=400,
)
st.plotly_chart(fig2, width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# Chart 3 — Compute vs Communication stacked bar
# ────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Chart 3 — Compute vs. Communication Time</div>',
            unsafe_allow_html=True)

figdf = df[["config", "mean_compute_s", "mean_comm_s"]].copy()
figdf["mean_compute_ms"] = figdf["mean_compute_s"] * 1000
figdf["mean_comm_ms"]    = figdf["mean_comm_s"]    * 1000

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name="Compute",
    x=figdf["config"],
    y=figdf["mean_compute_ms"],
    marker_color="#58a6ff",
))
fig3.add_trace(go.Bar(
    name="Communication (MPI)",
    x=figdf["config"],
    y=figdf["mean_comm_ms"],
    marker_color="#f78166",
))
fig3.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    title="Compute vs. Communication Time Breakdown (ms)",
    xaxis_title="Configuration",
    yaxis_title="Time (ms)",
    height=420,
)
st.plotly_chart(fig3, width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# Chart 4 — Graph preview via NetworkX
# ────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Chart 4 — Graph Preview (Sample)</div>',
            unsafe_allow_html=True)

@st.cache_data(ttl=120)
def build_sample_graph(path: str, n: int):
    if not os.path.isfile(path):
        return None, None
    G = nx.Graph()
    count = 0
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            G.add_edge(u, v)
            count += 1
            if G.number_of_nodes() >= n:
                break
    return G, nx.spring_layout(G, seed=42, k=0.5)

G_sample, pos = build_sample_graph(graph_sample_path, sample_size)

if G_sample is None or G_sample.number_of_nodes() == 0:
    st.info(
        "No graph file found for preview. "
        "Set the graph path in the sidebar or download `roadNet-CA.txt`."
    )
else:
    edge_x, edge_y = [], []
    for u, v in G_sample.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G_sample.nodes()]
    node_y = [pos[n][1] for n in G_sample.nodes()]
    degrees = [d for _, d in G_sample.degree()]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="#30363d"),
        hoverinfo="none",
    ))
    fig4.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            size=[3 + d * 0.8 for d in degrees],
            color=degrees,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Degree", tickfont=dict(color="#c9d1d9")),
            line=dict(width=0.5, color="#0d1117"),
        ),
        text=[f"v={n} deg={d}" for n, d in zip(G_sample.nodes(), degrees)],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Graph Sample ({G_sample.number_of_nodes()} vertices, "
              f"{G_sample.number_of_edges()} edges)",
        showlegend=False,
        height=520,
    )
    fig4.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig4.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    st.plotly_chart(fig4, width="stretch")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hybrid Parallel BFS · MPI.jl + Base.Threads · "
    "Graph: roadNet-CA (SNAP Stanford) · Dashboard: Streamlit + Plotly"
)
