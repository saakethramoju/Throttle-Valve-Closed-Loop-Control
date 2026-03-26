from .Actuators import saturation
from .Filters import low_pass_filter

class PID:
    """
    Discrete PID controller with output saturation, simple anti-windup,
    and filtered derivative-on-measurement.

    This controller is intended for sampled-time control of a scalar signal:

        error = target - measurement

    The control law is:

        u_unsat = u_bias + Kp * error + Ki * integral - Kd * d(measurement)/dt

    where the derivative is taken on the measurement instead of the error.
    This avoids derivative kick when the target changes suddenly. The raw
    measurement derivative is then low-pass filtered before use in the control
    law to reduce sensitivity to numerical noise, sensor noise, and small
    plant oscillations.

    Features
    --------
    - Proportional, integral, and derivative control
    - Derivative on measurement
    - First-order low-pass filtering of the derivative estimate
    - Output saturation between u_min and u_max
    - Simple anti-windup by freezing the integral when saturated and the error
      would push the controller farther into saturation
    - Optional constant bias term u_bias

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain. This multiplies the filtered derivative of the
        measurement.
    u_min : float
        Minimum allowable controller output.
    u_max : float
        Maximum allowable controller output.
    u_bias : float, optional
        Constant bias added to the controller output. Default is 0.0.
    tau_d : float, optional
        Time constant for the first-order low-pass filter applied to the raw
        measurement derivative. Larger values give more smoothing but reduce
        derivative responsiveness. Must be positive for meaningful filtering.
        Default is 0.05.

    Attributes
    ----------
    Kp, Ki, Kd : float
        Controller gains.
    u_min, u_max : float
        Output saturation limits.
    u_bias : float
        Constant bias term.
    tau_d : float
        Derivative filter time constant.
    error : float or None
        Most recent control error, target - measurement.
    integral : float
        Current integral state.
    derivative : float
        Most recent filtered derivative estimate of d(measurement)/dt.
    prev_measurement : float or None
        Previous measurement sample, used to compute the raw derivative.
    u_unsat : float or None
        Most recent controller output before saturation.
    u_sat : float or None
        Most recent controller output after saturation.

    Notes
    -----
    The derivative term is applied as:

        -Kd * derivative

    because increasing measurement implies the process variable is already
    moving upward, so the derivative term acts to oppose that motion.

    The anti-windup strategy used here is intentionally simple. It prevents the
    integral from growing when the controller is saturated and the current
    error would drive it even farther into the same saturation limit.

    This class assumes the helper functions `low_pass_filter(...)` and
    `saturation(...)` already exist in the surrounding codebase.
    """

    def __init__(self, Kp, Ki, Kd, u_min, u_max, u_bias=0.0, tau_d=0.05):
        Kp = float(Kp)
        Ki = float(Ki)
        Kd = float(Kd)
        u_min = float(u_min)
        u_max = float(u_max)
        u_bias = float(u_bias)
        tau_d = float(tau_d)

        if u_min > u_max:
            raise ValueError(
                f"u_min must be <= u_max. Got u_min={u_min}, u_max={u_max}."
            )

        if tau_d < 0.0:
            raise ValueError(f"tau_d must be >= 0. Got tau_d={tau_d}.")

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.u_min = u_min
        self.u_max = u_max
        self.u_bias = u_bias
        self.tau_d = tau_d

        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None

        self.u_unsat = None
        self.u_sat = None

    def reset(self, measurement=None):
        """
        Reset the controller internal state.

        Parameters
        ----------
        measurement : float or None, optional
            Current measured process value. If provided, it is stored as the
            previous measurement so that the next update does not produce a
            startup derivative spike. If omitted, the next derivative estimate
            is initialized from zero.
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
            Controller timestep in seconds. Must be positive.

        Returns
        -------
        u_sat : float
            Saturated controller output.
        error : float
            Current control error, target - measurement.
        integral : float
            Current integral state after anti-windup logic.
        derivative : float
            Current filtered derivative estimate of d(measurement)/dt.

        Notes
        -----
        The raw measurement derivative is computed as:

            d_meas_dt = (measurement - prev_measurement) / dt

        and then filtered with a first-order low-pass filter:

            derivative = low_pass_filter(previous_derivative, d_meas_dt, dt, tau_d)

        The integral is only accepted if the controller is not saturated in a
        direction that the present error would worsen.
        """
        dt = float(dt)
        target = float(target)
        measurement = float(measurement)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got dt={dt}.")

        error = target - measurement
        self.error = error

        # Derivative on measurement to avoid derivative kick from setpoint steps.
        if self.prev_measurement is None:
            d_meas_dt = 0.0
        else:
            d_meas_dt = (measurement - self.prev_measurement) / dt

        # Low-pass filter the raw derivative estimate.
        if self.tau_d == 0.0:
            self.derivative = d_meas_dt
        else:
            self.derivative = low_pass_filter(
                self.derivative,
                d_meas_dt,
                dt,
                self.tau_d,
            )

        # Tentative integral update.
        integral_candidate = self.integral + error * dt

        # Unsaturated controller output.
        u_unsat = (
            self.u_bias
            + self.Kp * error
            + self.Ki * integral_candidate
            - self.Kd * self.derivative
        )
        self.u_unsat = u_unsat

        # Apply output saturation.
        u_sat = saturation(u_unsat, self.u_min, self.u_max)
        self.u_sat = u_sat

        # Freeze integral if saturated and error would push farther into saturation.
        pushing_further_high = (u_unsat > self.u_max) and (error > 0.0)
        pushing_further_low = (u_unsat < self.u_min) and (error < 0.0)

        if not (pushing_further_high or pushing_further_low):
            self.integral = integral_candidate

        self.prev_measurement = measurement

        return u_sat, self.error, self.integral, self.derivative
    

    
'''
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
'''