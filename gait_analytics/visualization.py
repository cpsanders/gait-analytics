import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
import polars as pl

def plot_data(
    df: pl.DataFrame,
    accel_x_column: str = "accel_x",
    accel_y_column: str = "accel_y",
    accel_z_column: str = "accel_z",
    accel_magnitude_column: str = "accel_magnitude",
    accel_magnitude_smoothed_column: str = "accel_magnitude_smoothed",
    speed_column: str = "speed_mps",
    target_speed_column: str = "target_speed_mps",
    target_speed_column_smoothed: str = "target_speed_mps_smoothed",
    cadence_column: str = "cadence_steps_per_minute",
    lead_sec: int = 60,
):
    # Updated to 4 rows
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        subplot_titles=(
            "Raw Accelerometer (X, Y, Z)", 
            "Gait Intensity (Magnitude)",
            "Cadence (Steps per Minute)",
            f"Velocity Trend",
        )
    )

    # --- Subplot 1: Raw Components ---
    colors_raw = ["#636EFA", "#EF553B", "#00CC96"] 
    for i, axis in enumerate([accel_x_column, accel_y_column, accel_z_column]):
        fig.add_trace(
            go.Scatter(y=df[axis], name=axis, opacity=0.4, line=dict(color=colors_raw[i])), 
            row=1, col=1
        )

    # --- Subplot 2: Magnitudes ---
    fig.add_trace(
        go.Scatter(y=df[accel_magnitude_column], name="Raw Mag", opacity=0.3, line=dict(color="orange")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=df[accel_magnitude_smoothed_column], name="Smooth Mag", line=dict(color="yellow", width=2)),
        row=2, col=1
    )

    # --- Subplot 3: Cadence ---
    fig.add_trace(
        go.Scatter(
            y=df[cadence_column],
            name="Cadence (SPM)", 
            line=dict(color="#00D4FF", width=2), # Bright Cyan
            fill="tozeroy", # Shaded area under cadence looks great for step data
            fillcolor="rgba(0, 212, 255, 0.1)"
        ), 
        row=3, col=1
    )

    # --- Subplot 4: Prediction Target ---
    fig.add_trace(
        go.Scatter(y=df[speed_column], name="Current Speed (Raw)", opacity=0.3, line=dict(color="gray")),
        row=4, col=1
    )
    # fig.add_trace(
    #     go.Scatter(y=df[target_speed_column], name="Target Speed (Raw)", opacity=0.3, line=dict(color="magenta")),
    #     row=4, col=1
    # )
    fig.add_trace(
        go.Scatter(
            y=df[target_speed_column_smoothed],
            name=f"Smooth Target ",
            line=dict(color="#FF4500", width=3)
        ), 
        row=4, col=1
    )

    # --- Layout Improvements ---
    fig.update_layout(
        height=1200, # Increased height for the extra row
        title_text="Gait & Cadence Analysis Dashboard",
        showlegend=True,
        template="plotly_dark",
        hovermode="x unified"
    )
    
    # Update Y-axes
    fig.update_yaxes(title_text="G-Force", row=1, col=1)
    fig.update_yaxes(title_text="G-Force", row=2, col=1)
    fig.update_yaxes(title_text="SPM", row=3, col=1) # Steps Per Minute
    fig.update_yaxes(title_text="m/s", row=4, col=1)
    
    fig.show()