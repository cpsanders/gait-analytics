import polars as pl


def filter_stationary_timestamps(df: pl.DataFrame, speed_column: str = "speed_mps") -> pl.DataFrame:
    """Remove rows where no motion was occurring (speed is 0)."""
    return df.filter(pl.col(speed_column) != 0.0)


def add_acceleration_magnitude(
    df: pl.DataFrame,
    accel_x_column: str = "accel_x",
    accel_y_column: str = "accel_y",
    accel_z_column: str = "accel_z",
) -> pl.DataFrame:
    """Add acceleration magnitude feature."""
    # Total acceleration = sqrt(x^2 + y^2 + z^2)
    return df.with_columns(
        accel_magnitude = (pl.col(accel_x_column)**2 + pl.col(accel_y_column)**2 + pl.col(accel_z_column)**2).sqrt()
    )


def add_smoothed_column(
    df: pl.DataFrame,
    column_name: str,
    rolling_mean_window: int = 5,
    center = True,
    smoothed_column_name: str | None = None,
):
    """
    Add a rolling mean smoothed column.

    Args:
        df: input dataframe to process
        column_name: column to smooth
        rolling_mean_window: row window over which to calculate the smoothing mean
        center: flag to center the rolling mean window
    """
    if smoothed_column_name is None:
        smoothed_column_name = f"{column_name}_smoothed"

    # Smooth the signal using a rolling mean. Keep some noise, but remove 'jitters'.
    return df.with_columns(
        pl.col(column_name).rolling_mean(window_size=rolling_mean_window, center=center).alias(smoothed_column_name)
    )


def add_step_cadence(
    df: pl.DataFrame,
    step_g_force_threshold: float = 1.2,
    cadence_window_sec: int = 3,
    hz: int = 50,
    accel_magnitude_column: str = "accel_magnitude"
) -> pl.DataFrame:
    """
    Add column for cadence in steps per minute.

    Args:
        df: input dataframe to process
        hz: number of measurements per second
        step_g_force_threshold: G-Force threshold for classification of a "step"
        cadence_window_sec: number of seconds over which to calculate the cadence window
    """
    # Look for rows that are 'local peaks' and above the step_g_force_threshold
    df = df.with_columns(
        is_step = pl.when(
            (pl.col(accel_magnitude_column) > pl.col(accel_magnitude_column).shift(1)) &
            (pl.col(accel_magnitude_column) > pl.col(accel_magnitude_column).shift(-1)) &
            (pl.col(accel_magnitude_column) > step_g_force_threshold) # anything less than 1.2 is just a wobble
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Boolean)
    )

    # Rolling sum of steps over cadence_window_sec seconds, then scale to 60 seconds
    return df.with_columns(
        cadence_steps_per_minute = (
            pl.col("is_step")
            .rolling_sum(window_size=cadence_window_sec * hz) * (60 / cadence_window_sec)
        )
    )


def add_future_speed_target(
    df: pl.DataFrame,
    lead_sec: int = 60,
    hz: int = 50,
    speed_column: str = "speed_mps",
    target_speed_column: str | None = None,
) -> pl.DataFrame:
    """
    Add a future speed target column lead_sec in advance.

    Args:
        df: dataframe to be processed
        lead_sec: number of seconds out for target
        hz: number of measurements per second
        speed_column: column to use for speed measurements
        target_speed_column: target speed column name
    """
    if target_speed_column is None:
        target_speed_column = f"{speed_column}_target"

    # Shift the pace column backward by (lead_sec * hz) samples
    shift_samples = lead_sec * hz
    return df.with_columns(pl.col(speed_column).shift(-shift_samples).alias(target_speed_column))


def process_gait_data(
    df: pl.DataFrame,
    lead_sec: int = 60,
    hz: int = 50,
    rolling_mean_window: int = 5,
    step_g_force_threshold: float = 1.2,
    cadence_window_sec: int = 3,
    speed_column: str = "speed_mps",
    accel_x_column: str = "accel_x",
    accel_y_column: str = "accel_y",
    accel_z_column: str = "accel_z",
    accel_magnitude_column: str = "accel_magnitude",
) -> pl.DataFrame:
    """
    Process raw gait data.

    Args:
        df: input dataframe to process
        lead_sec: the number of seconds to look in the future for the velocity target
        hz: measurement spec of the sensor
        rolling_mean_window: row window size over which to calculate rolling means
        step_threshold: G-Force threshold for classification of a "step"
        cadence_window_sec: number of seconds over which to calculate the cadence window
        speed_column: column in the data recording speed
        accel_x_column: accelerometer x column
        accel_y_column: accelerometer y column
        accel_z_column: accelerometer z column
        accel_magnitude_column: acceleration magnitude column

    Returns:
        Processed dataframe.
    """
    df = filter_stationary_timestamps(df, speed_column=speed_column)

    df = add_acceleration_magnitude(
        df,
        accel_x_column=accel_x_column,
        accel_y_column=accel_y_column,
        accel_z_column=accel_z_column,
    )

    df = add_step_cadence(
        df,
        step_g_force_threshold=step_g_force_threshold,
        cadence_window_sec=cadence_window_sec,
        accel_magnitude_column=accel_magnitude_column,
    )

    df = add_smoothed_column(
        df,
        column_name=accel_magnitude_column,
        rolling_mean_window=rolling_mean_window,
        center=True,
    )

    target_speed_column = f"target_{speed_column}"
    df = add_future_speed_target(
        df,
        lead_sec=lead_sec,
        hz=hz,
        speed_column=speed_column,
        target_speed_column=target_speed_column,
    )

    df = add_smoothed_column(
        df,
        column_name=target_speed_column,
        rolling_mean_window=rolling_mean_window * hz,
        center=True,
    )

    # Drop nulls created by the shift at the end of the file
    return df.drop_nulls()
