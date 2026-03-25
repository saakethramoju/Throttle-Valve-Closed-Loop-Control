"""
Actuator utilities and simple actuator models.

The actuator is useful when a controller computes a desired command, but the
physical device cannot instantaneously jump to that command. Instead, the
actuator:

1. Enforces hard minimum and maximum output values.
2. Enforces a maximum rate of change per second.

This makes closed-loop simulations more realistic than directly writing the
controller command into the plant.
"""

from .ControlUtilities import saturation


class TestActuator:
    """
    Simple rate-limited actuator with hard output bounds.

    This class represents a first-order-ish actuator model for situations where
    a commanded value cannot be achieved instantaneously. The actuator stores
    an internal state `value`, representing the actual physical actuator output.

    At each update step, the actuator:

    1. Saturates the incoming command to the allowable position range.
    2. Computes the difference between the command and the current actuator value.
    3. Limits that difference so the actuator cannot move faster than `max_rate`.
    4. Updates the internal actuator value.
    5. Saturates the final value again to guarantee it remains in bounds.

    This is especially useful in control simulations where the controller
    outputs a desired command, but the plant should receive a more realistic
    actuator response.

    Parameters
    ----------
    initial_value : float
        Initial actuator output value.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    max_rate : float
        Maximum actuator rate, in output-units per second.
        For a valve CdA actuator, this would typically be in m^2/s.

    Attributes
    ----------
    value : float
        Current actuator output value.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    max_rate : float
        Maximum actuator slew rate.

    Raises
    ------
    ValueError
        If:
        - min_value > max_value
        - max_rate < 0
        - initial_value is outside [min_value, max_value]

    Notes
    -----
    If the controller command is already within the actuator's achievable
    change for the current timestep, the actuator will exactly reach the command.

    If the command is too far away, the actuator will move only by the maximum
    allowed increment:

        max_step = max_rate * dt

    so that the actuator respects its finite rate limit.

    Mathematical Form
    -----------------
    Let:

    - u_cmd be the commanded actuator value
    - u_k be the current actuator value
    - dt be the timestep
    - r_max be the maximum rate

    Then the update law is:

        u_cmd_sat = sat(u_cmd, u_min, u_max)

        max_step = r_max * dt

        delta = sat(u_cmd_sat - u_k, -max_step, +max_step)

        u_{k+1} = sat(u_k + delta, u_min, u_max)

    This produces bounded, rate-limited motion toward the commanded value.

    Examples
    --------
    >>> actuator = TestActuator(
    ...     initial_value=0.5e-4,
    ...     min_value=0.2e-4,
    ...     max_value=1.0e-4,
    ...     max_rate=1.0e-5,
    ... )
    >>> actuator.update(command=0.9e-4, dt=0.01)
    5.01e-05

    In that example, the actuator does not jump directly to 0.9e-4. Instead,
    it moves only by the maximum amount allowed during that timestep.
    """

    def __init__(self, initial_value, min_value, max_value, max_rate):
        min_value = float(min_value)
        max_value = float(max_value)
        initial_value = float(initial_value)
        max_rate = float(max_rate)

        if min_value > max_value:
            raise ValueError(
                f"min_value must be <= max_value. Got min_value={min_value}, max_value={max_value}."
            )

        if max_rate < 0.0:
            raise ValueError(f"max_rate must be nonnegative. Got max_rate={max_rate}.")

        if not (min_value <= initial_value <= max_value):
            raise ValueError(
                "initial_value must lie within [min_value, max_value]. "
                f"Got initial_value={initial_value}, min_value={min_value}, max_value={max_value}."
            )

        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.max_rate = max_rate

    def update(self, command, dt):
        """
        Advance the actuator by one timestep toward the commanded value.

        Parameters
        ----------
        command : float
            Desired actuator value from the controller.
        dt : float
            Simulation timestep in seconds. Must be nonnegative.

        Returns
        -------
        float
            Updated actuator output value after applying:
            - command saturation
            - slew-rate limiting
            - final output saturation

        Raises
        ------
        ValueError
            If dt < 0.

        Notes
        -----
        The actuator does not necessarily reach the commanded value in one
        timestep. Instead, it moves toward the command no faster than:

            max_rate * dt

        per update call.

        Behavior summary:
        - If `command` exceeds position limits, it is clipped.
        - If the required motion exceeds the allowed rate, it is reduced.
        - The internal actuator state `self.value` is updated and returned.
        """
        dt = float(dt)
        command = float(command)

        if dt < 0.0:
            raise ValueError(f"dt must be nonnegative. Got dt={dt}.")

        command = saturation(command, self.min_value, self.max_value)

        max_step = self.max_rate * dt
        delta = command - self.value
        delta = saturation(delta, -max_step, max_step)

        self.value += delta
        self.value = saturation(self.value, self.min_value, self.max_value)

        return self.value

    def reset(self, value=None):
        """
        Reset the actuator state.

        Parameters
        ----------
        value : float or None, optional
            Value to reset the actuator to. If None, the actuator resets to
            `min_value`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided reset value lies outside the actuator bounds.
        """
        if value is None:
            value = self.min_value

        value = float(value)

        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                "Reset value must lie within [min_value, max_value]. "
                f"Got value={value}, min_value={self.min_value}, max_value={self.max_value}."
            )

        self.value = value

    @property
    def normalized_value(self):
        """
        Return actuator position normalized to [0, 1].

        Returns
        -------
        float
            Normalized actuator value:
            - 0 corresponds to min_value
            - 1 corresponds to max_value

        Notes
        -----
        This can be useful for plotting actuator travel as a fraction of full
        stroke instead of in physical units.
        """
        span = self.max_value - self.min_value
        if span == 0.0:
            return 0.0
        return (self.value - self.min_value) / span

    def __repr__(self):
        """
        Return a developer-friendly string representation of the actuator.
        """
        return (
            f"{self.__class__.__name__}("
            f"value={self.value}, "
            f"min_value={self.min_value}, "
            f"max_value={self.max_value}, "
            f"max_rate={self.max_rate})"
        )