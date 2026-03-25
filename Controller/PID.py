from .ControlUtilities import saturation


class PID:
    """
    Discrete-time PID controller with output saturation and simple anti-windup.

    This controller computes a control command from the difference between a
    target value and a measured value:

        error = target - measurement

    The control law is

        u = u_bias + Kp * error + Ki * integral - Kd * d(measurement)/dt

    where:
    - Kp is the proportional gain
    - Ki is the integral gain
    - Kd is the derivative gain
    - u_bias is a nominal steady-state control value

    Notes
    -----
    1. Derivative-on-measurement is used instead of derivative-on-error.
       This helps reduce derivative kick when the target changes suddenly.

    2. Output saturation is applied to keep the control command within the
       allowed range [u_min, u_max].

    3. A simple conditional-integration anti-windup scheme is used.
       In plain terms:
       - if the controller is already at its maximum output and the error is
         asking for even more positive output, do not keep growing the integral
       - if the controller is already at its minimum output and the error is
         asking for even more negative output, do not keep growing the integral

       This prevents the integral term from "winding up" while the actuator is
       stuck at a limit.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.
    u_min : float
        Minimum allowed controller output.
    u_max : float
        Maximum allowed controller output.
    u_bias : float, optional
        Nominal control value about which the PID makes corrections.
        Default is 0.0.

    Attributes
    ----------
    Kp, Ki, Kd : float
        PID gains.
    u_min, u_max : float
        Output limits.
    u_bias : float
        Bias term added to the PID output.
    error : float or None
        Most recent control error.
    integral : float
        Current integral state.
    derivative : float
        Most recent derivative term, stored as d(measurement)/dt.
    prev_measurement : float or None
        Previous measurement used for derivative calculation.
    u_unsat : float or None
        Most recent controller output before saturation.
    u_sat : float or None
        Most recent controller output after saturation.

    Examples
    --------
    >>> pid = PID(Kp=1.0, Ki=0.5, Kd=0.1, u_min=0.0, u_max=10.0, u_bias=5.0)
    >>> u, e, i, d = pid.update(target=2.0, measurement=1.8, dt=0.01)
    """

    def __init__(self, Kp, Ki, Kd, u_min, u_max, u_bias=0.0):
        Kp = float(Kp)
        Ki = float(Ki)
        Kd = float(Kd)
        u_min = float(u_min)
        u_max = float(u_max)
        u_bias = float(u_bias)

        if u_min > u_max:
            raise ValueError(
                f"u_min must be <= u_max. Got u_min={u_min}, u_max={u_max}."
            )

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.u_min = u_min
        self.u_max = u_max
        self.u_bias = u_bias

        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None

        self.u_unsat = None
        self.u_sat = None

    def reset(self, measurement=None):
        """
        Reset the controller state.

        Parameters
        ----------
        measurement : float or None, optional
            If provided, this value is stored as the previous measurement so
            the next derivative estimate does not see a large artificial jump.
            If None, the previous measurement is cleared.

        Returns
        -------
        None
        """
        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None if measurement is None else float(measurement)
        self.u_unsat = None
        self.u_sat = None

    def update(self, target, measurement, dt):
        """
        Advance the PID controller by one timestep.

        Parameters
        ----------
        target : float
            Desired setpoint.
        measurement : float
            Current measured process value.
        dt : float
            Timestep in seconds. Must be positive.

        Returns
        -------
        tuple
            A 4-tuple:

            (u_sat, error, integral, derivative)

            where:
            - u_sat is the saturated controller output
            - error is the current control error
            - integral is the current integral state
            - derivative is d(measurement)/dt

        Raises
        ------
        ValueError
            If dt <= 0.

        Notes
        -----
        Anti-windup behavior:
        If the controller is already trying to push past one of its output
        limits, and the current error would push it even farther into that
        limit, the integral term is not updated on this step.

        In simpler terms:
        - at max output, do not keep integrating positive error
        - at min output, do not keep integrating negative error
        """
        dt = float(dt)
        target = float(target)
        measurement = float(measurement)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got dt={dt}.")

        error = target - measurement
        self.error = error

        # Derivative on measurement reduces derivative kick from sudden setpoint changes.
        if self.prev_measurement is None:
            d_meas_dt = 0.0
        else:
            d_meas_dt = (measurement - self.prev_measurement) / dt

        self.derivative = d_meas_dt

        # Tentatively update the integral, then decide whether to keep it.
        integral_candidate = self.integral + error * dt

        # Raw controller output before output limiting.
        u_unsat = (
            self.u_bias
            + self.Kp * error
            + self.Ki * integral_candidate
            - self.Kd * d_meas_dt
        )
        self.u_unsat = u_unsat

        # Final controller output after applying bounds.
        u_sat = saturation(u_unsat, self.u_min, self.u_max)
        self.u_sat = u_sat

        # Simple anti-windup:
        # If the controller is already beyond a limit, and the error is trying
        # to push it even farther into that same limit, freeze the integral.
        pushing_further_high = (u_unsat > self.u_max) and (error > 0.0)
        pushing_further_low = (u_unsat < self.u_min) and (error < 0.0)

        if not (pushing_further_high or pushing_further_low):
            self.integral = integral_candidate

        self.prev_measurement = measurement

        return u_sat, self.error, self.integral, self.derivative

    def __repr__(self):
        """
        Return a developer-friendly representation of the controller state.
        """
        return (
            f"{self.__class__.__name__}("
            f"Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
            f"u_min={self.u_min}, u_max={self.u_max}, u_bias={self.u_bias}, "
            f"error={self.error}, integral={self.integral}, derivative={self.derivative})"
        )