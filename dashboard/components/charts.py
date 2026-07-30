"""
CollideX Dashboard — Chart Components
=======================================
All Plotly chart builders. Returns go.Figure objects.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from config import RISK_COLORS, PLOTLY_LAYOUT_DEFAULTS


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply the global dark layout defaults to a figure."""
    layout = dict(**PLOTLY_LAYOUT_DEFAULTS)
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color="#E8F4FD"))
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def risk_donut(risk_counts: dict) -> go.Figure:
    """Donut chart showing risk class distribution."""
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [RISK_COLORS.get(lbl, "#8BA3C7") for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#050A14", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#E8F4FD"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:11px'>Objects</span>",
        x=0.5, y=0.5, font=dict(size=18, color="#E8F4FD"),
        showarrow=False
    )
    return _apply_layout(fig, "Risk Class Distribution",
                         height=320, showlegend=True)


def collision_prob_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of collision probabilities."""
    fig = go.Figure(go.Histogram(
        x=df["collision_probability"],
        nbinsx=50,
        marker=dict(
            color=df["collision_probability"],
            colorscale=[[0, "#00FF9F"], [0.35, "#FFD700"], [0.65, "#FF3860"], [1, "#FF3860"]],
            line=dict(color="#050A14", width=0.5)
        ),
        hovertemplate="Probability: %{x:.3f}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=0.35, line_dash="dash", line_color="#FFD700",
                  annotation_text="Medium", annotation_font_color="#FFD700")
    fig.add_vline(x=0.65, line_dash="dash", line_color="#FF3860",
                  annotation_text="High", annotation_font_color="#FF3860")
    return _apply_layout(fig, "Collision Probability Distribution",
                         xaxis_title="Collision Probability",
                         yaxis_title="Number of Objects",
                         height=320)


def top_risk_bar(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-N highest risk objects."""
    top = df.nlargest(n, "collision_probability").copy()
    top["label"] = top["satellite_name"].fillna("UNKNOWN").str[:20]
    colors = [RISK_COLORS.get(r, "#8BA3C7") for r in top["risk_label"]]

    fig = go.Figure(go.Bar(
        y=top["label"][::-1],
        x=top["collision_probability"][::-1],
        orientation="h",
        marker=dict(color=colors[::-1],
                    line=dict(color="#050A14", width=0.5)),
        text=[f"{v:.3f}" for v in top["collision_probability"][::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#8BA3C7"),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>",
    ))
    return _apply_layout(fig, f"Top {n} Highest Risk Objects",
                         xaxis_title="Collision Probability",
                         height=max(320, n * 22),
                         xaxis=dict(range=[0, 1.1], **PLOTLY_LAYOUT_DEFAULTS["xaxis"]))


def altitude_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: Altitude vs Collision Probability coloured by risk."""
    color_map = {"Low": "#00FF9F", "Medium": "#FFD700", "High": "#FF3860"}
    fig = go.Figure()
    for label, group in df.groupby("risk_label"):
        fig.add_trace(go.Scatter(
            x=group["altitude_km"],
            y=group["collision_probability"],
            mode="markers",
            name=label,
            marker=dict(
                color=color_map.get(label, "#8BA3C7"),
                size=5,
                opacity=0.7,
                line=dict(color="#050A14", width=0.5),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Alt: %{x:.1f} km<br>Prob: %{y:.4f}<extra></extra>"
            ),
            customdata=df.loc[group.index, ["satellite_name"]].fillna("UNKNOWN"),
        ))
    return _apply_layout(fig, "Altitude vs Collision Probability",
                         xaxis_title="Altitude (km)",
                         yaxis_title="Collision Probability",
                         height=340)


def velocity_altitude_heatmap(df: pd.DataFrame) -> go.Figure:
    """2D density: velocity vs altitude."""
    fig = go.Figure(go.Histogram2dContour(
        x=df["altitude_km"].clip(0, 2000),
        y=df["velocity_mag_km_s"],
        colorscale=[[0, "#050A14"], [0.3, "#0A2040"],
                    [0.6, "#1E6FFF"], [1.0, "#00D4FF"]],
        contours=dict(showlabels=False),
        hovertemplate="Alt: %{x:.0f} km<br>Vel: %{y:.2f} km/s<extra></extra>",
        line=dict(width=0),
    ))
    return _apply_layout(fig, "Velocity-Altitude Density Map",
                         xaxis_title="Altitude (km)",
                         yaxis_title="Velocity (km/s)",
                         height=340)


def risk_time_series(df: pd.DataFrame) -> go.Figure:
    """NORAD sorted line with collision probability (acts as timeline proxy)."""
    sorted_df = df.sort_values("norad_id").reset_index(drop=True)
    sampled   = sorted_df.iloc[::max(1, len(sorted_df)//500)]  # downsample for performance

    colors = [RISK_COLORS.get(r, "#8BA3C7") for r in sampled["risk_label"]]
    fig = go.Figure()
    for label, col in [("Low", "#00FF9F"), ("Medium", "#FFD700"), ("High", "#FF3860")]:
        mask = sampled["risk_label"] == label
        fig.add_trace(go.Scatter(
            x=sampled.loc[mask, "norad_id"],
            y=sampled.loc[mask, "collision_probability"],
            mode="markers",
            name=label,
            marker=dict(color=col, size=3, opacity=0.6),
            hovertemplate="NORAD %{x}<br>Prob: %{y:.4f}<extra></extra>",
        ))
    return _apply_layout(fig, "Collision Risk by NORAD Catalog ID",
                         xaxis_title="NORAD Catalog ID",
                         yaxis_title="Collision Probability",
                         height=320)


def gauge_chart(value: float, title: str, max_val: float = 1.0) -> go.Figure:
    """Single gauge for a probability/score."""
    pct = value / max_val
    if pct >= 0.65:
        color = "#FF3860"
        risk  = "HIGH RISK"
    elif pct >= 0.35:
        color = "#FFD700"
        risk  = "MEDIUM RISK"
    else:
        color = "#00FF9F"
        risk  = "LOW RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text=f"{title}<br><span style='font-size:14px;color:{color};'>{risk}</span>",
                   font=dict(size=14, color="#E8F4FD")),
        number=dict(font=dict(size=32, color=color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#4A6A8A",
                      tickfont=dict(color="#4A6A8A", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(10,22,40,0.8)",
            borderwidth=1,
            bordercolor="#1A3A5C",
            steps=[
                dict(range=[0, 0.35 * max_val], color="rgba(0,255,159,0.1)"),
                dict(range=[0.35 * max_val, 0.65 * max_val], color="rgba(255,215,0,0.1)"),
                dict(range=[0.65 * max_val, max_val], color="rgba(255,56,96,0.1)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=value,
            )
        )
    ))
    return _apply_layout(fig, height=260)


def position_3d_scatter(df: pd.DataFrame, n_sample: int = 1000) -> go.Figure:
    """3D scatter of future positions coloured by risk."""
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    color_map = {"Low": "#00FF9F", "Medium": "#FFD700", "High": "#FF3860"}

    fig = go.Figure()
    for label, col in color_map.items():
        mask = sample["risk_label"] == label
        sub  = sample[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["future_x_km"], y=sub["future_y_km"], z=sub["future_z_km"],
            mode="markers",
            name=label,
            marker=dict(size=2, color=col, opacity=0.7),
            hovertemplate="<b>%{customdata}</b><br>X:%{x:.0f} Y:%{y:.0f} Z:%{z:.0f}<extra></extra>",
            customdata=sub["satellite_name"].fillna("?"),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        scene=dict(
            xaxis=dict(backgroundcolor="#050A14", gridcolor="#1A3A5C",
                       showbackground=True, title="X (km)"),
            yaxis=dict(backgroundcolor="#050A14", gridcolor="#1A3A5C",
                       showbackground=True, title="Y (km)"),
            zaxis=dict(backgroundcolor="#050A14", gridcolor="#1A3A5C",
                       showbackground=True, title="Z (km)"),
            bgcolor="#050A14",
        ),
        title=dict(text="3D Future Position Coordinates",
                   font=dict(size=15, color="#E8F4FD")),
        height=480,
    )
    return fig


def metrics_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of model metrics."""
    key_metrics = [
        "Accuracy", "Balanced Accuracy", "ROC-AUC (macro OVR)",
        "F1-Score (macro)", "Precision (macro)", "Recall (macro)"
    ]
    filtered = metrics_df[metrics_df["Metric"].isin(key_metrics)].copy()
    filtered["Value"] = filtered["Value"].astype(float) * 100

    fig = go.Figure(go.Bar(
        y=filtered["Metric"],
        x=filtered["Value"],
        orientation="h",
        marker=dict(
            color=filtered["Value"],
            colorscale=[[0, "#1E6FFF"], [0.95, "#00D4FF"], [1.0, "#00FF9F"]],
            cmin=90, cmax=100,
            line=dict(color="#050A14", width=0.5),
        ),
        text=[f"{v:.4f}%" for v in filtered["Value"]],
        textposition="outside",
        textfont=dict(size=11, color="#8BA3C7"),
        hovertemplate="<b>%{y}</b><br>%{x:.6f}%<extra></extra>",
    ))
    return _apply_layout(fig, "Model Performance Metrics (%)",
                         xaxis_title="Score (%)",
                         xaxis=dict(range=[98, 101], **PLOTLY_LAYOUT_DEFAULTS["xaxis"]),
                         height=300)
