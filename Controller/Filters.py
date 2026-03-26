import numpy as np


def apply_error_deadband(target: float, measurement: float, tolerance: float) -> float:
    """
    Return an effective target that disables controller action when the error
    magnitude is below a specified tolerance.
    """
    error = target - measurement
    if abs(error) < tolerance:
        return measurement
    return target


def saturation(x, x_min, x_max):
    """
    Clamp a scalar value between lower and upper bounds.

    Parameters
    ----------
    x : float
        Value to be clamped.
    x_min : float
        Lower allowable bound.
    x_max : float
        Upper allowable bound.

    Returns
    -------
    float
        The clamped value:
        - x_min if x < x_min
        - x_max if x > x_max
        - x otherwise

    Raises
    ------
    ValueError
        If x_min > x_max.

    Notes
    -----
    This is a standard saturation function used heavily in control systems.
    It is useful for representing physical limits such as:

    - valve minimum / maximum CdA
    - motor minimum / maximum angle
    - command voltage limits
    - bounded control effort

    Examples
    --------
    >>> saturation(5.0, 0.0, 10.0)
    5.0
    >>> saturation(-2.0, 0.0, 10.0)
    0.0
    >>> saturation(12.0, 0.0, 10.0)
    10.0
    """
    if x_min > x_max:
        raise ValueError(f"x_min must be <= x_max. Got x_min={x_min}, x_max={x_max}.")

    return max(x_min, min(x, x_max))




def low_pass_filter(prev_filtered, measurement, dt, tau):
    """
    First-order low-pass filter (exponential smoothing).

    Parameters
    ----------
    prev_filtered : float
        Previous filtered value (x_f[k-1]).
    measurement : float
        Current raw measurement (x[k]).
    dt : float
        Timestep (seconds).
    tau : float
        Filter time constant (seconds). Larger tau = more smoothing.

    Returns
    -------
    float
        Updated filtered value (x_f[k]).
    """
    alpha = dt / (tau + dt)
    return prev_filtered + alpha * (measurement - prev_filtered)
