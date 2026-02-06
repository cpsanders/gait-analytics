import pathlib
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


def add_smoothed_speed_target(
    df: pl.DataFrame,
    hz: int = 50,
    window_size_sec: int = 2,
    center: bool = True,
    speed_column: str = "speed_mps",
    target_speed_column: str | None = None,
) -> pl.DataFrame:
    """
    Add a smothed speed target column, using a rolling mean over window_size_sec seconds.

    When creating the speed target, we need to add smoothing to account for jitter/error in the speed measurements.

    Args:
        df: dataframe to be processed
        hz: number of measurements per second
        window_size_sec: window size in seconds for rolling mean
        center: flag to center the rolling mean window
    """
    if target_speed_column is None:
        target_speed_column = f"{speed_column}_target_smoothed"

    return df.with_columns(
        pl.col(speed_column).rolling_mean(window_size=window_size_sec*hz, center=center).alias(target_speed_column)
    )


def trim_n_seconds_from_start_and_end(
    df: pl.DataFrame,
    n_seconds: int = 10,
    hz: int = 50,
) -> pl.DataFrame:
    """
    Trim n seconds of data from the start and end of the dataframe.

    Args:
        df: dataframe to be processed
        n_seconds: number of seconds to trim from start and end
        hz: number of measurements per second
    """
    n_rows_to_trim = n_seconds * hz
    return df.slice(n_rows_to_trim, len(df) - 2 * n_rows_to_trim)


def process_gait_data(
    file_path: pathlib.Path,
    hz: int = 50,
    feature_rolling_mean_window: int = 5,
    target_rolling_mean_window: int = 2,
    step_g_force_threshold: float = 1.2,
    cadence_window_sec: int = 3,
    n_seconds_to_trim: int = 10,
    speed_column: str = "speed_mps",
    accel_x_column: str = "accel_x",
    accel_y_column: str = "accel_y",
    accel_z_column: str = "accel_z",
    accel_magnitude_column: str = "accel_magnitude",
    output_path: pathlib.Path | None = None,
) -> pl.DataFrame:
    """
    Process raw gait data.

    Args:
        file_path: path to the raw data file
        hz: measurement spec of the sensor
        feature_rolling_mean_window: row window size over which to calculate rolling means
        target_rolling_mean_window_sec: window size in seconds for smoothing the target speed
        step_threshold: G-Force threshold for classification of a "step"
        cadence_window_sec: number of seconds over which to calculate the cadence window
        speed_column: column in the data recording speed
        accel_x_column: accelerometer x column
        accel_y_column: accelerometer y column
        accel_z_column: accelerometer z column
        accel_magnitude_column: acceleration magnitude column
        output_path: path to save the processed data file

    Returns:
        Processed dataframe.
    """
    df = pl.read_csv(file_path)

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
        rolling_mean_window=feature_rolling_mean_window,
        center=True,
    )

    target_speed_column = f"target_{speed_column}_smoothed"
    df = add_smoothed_speed_target(
        df,
        hz=hz,
        window_size_sec=target_rolling_mean_window,
        center=True,
        speed_column=speed_column,
        target_speed_column=target_speed_column,
    )

    df = trim_n_seconds_from_start_and_end(
        df,
        n_seconds=n_seconds_to_trim,
        hz=hz,
    )

    # Drop nulls created by the shift at the end of the file
    df = df.drop_nulls()

    if output_path is not None:
        df.write_parquet(output_path)

    return df
