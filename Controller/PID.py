from .Actuators import saturation



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
class PID:
    """
    Discrete-time PID controller with optional output saturation and support for
    preview/commit style anti-windup.

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

    2. Output saturation may be applied on a per-call basis by passing
       u_min and u_max into preview() or update().

    3. The controller supports two operating modes:

       A) Standard mode via update():
          - computes the control output
          - immediately commits the candidate integral state
          - optionally applies simple internal anti-windup against PID-level
            output saturation

       B) Preview/commit mode:
          - preview() computes the control output and stores candidate state
            internally, but does not commit it
          - commit() finalizes the step, with optional control over whether the
            integral is allowed to update

       The preview/commit flow is useful for outer-loop anti-windup, such as
       when actuator saturation is detected only after multiple PID outputs are
       mixed and passed through actuator models.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.

    Attributes
    ----------
    Kp, Ki, Kd : float
        PID gains.
    error : float or None
        Most recent control error.
    integral : float
        Current committed integral state.
    derivative : float
        Most recent derivative term, stored as d(measurement)/dt.
    prev_measurement : float or None
        Previous measurement used for derivative calculation.
    u_unsat : float or None
        Most recent controller output before saturation.
    u_sat : float or None
        Most recent controller output after saturation.

    _preview_valid : bool
        True if preview() has been called and commit() may be used.
    _candidate_integral : float or None
        Candidate integral state from the most recent preview().
    _candidate_error : float or None
        Candidate error from the most recent preview().
    _candidate_derivative : float or None
        Candidate derivative term from the most recent preview().
    _candidate_measurement : float or None
        Candidate measurement from the most recent preview().
    _candidate_u_unsat : float or None
        Candidate unsaturated output from the most recent preview().
    _candidate_u_sat : float or None
        Candidate saturated output from the most recent preview().
    _candidate_internal_allow_integrate : bool or None
        Internal PID-level anti-windup decision from the most recent preview().

    Examples
    --------
    Standard one-shot update:

    >>> pid = PID(Kp=1.0, Ki=0.5, Kd=0.1)
    >>> u, e, i, d = pid.update(
    ...     target=2.0,
    ...     measurement=1.8,
    ...     dt=0.01,
    ...     u_min=0.0,
    ...     u_max=10.0,
    ...     u_bias=5.0,
    ... )

    Preview/commit workflow:

    >>> pid = PID(Kp=1.0, Ki=0.5, Kd=0.1)
    >>> u, e, i, d = pid.preview(
    ...     target=2.0,
    ...     measurement=1.8,
    ...     dt=0.01,
    ...     u_min=0.0,
    ...     u_max=10.0,
    ...     u_bias=5.0,
    ... )
    >>> pid.commit(allow_integrate=True)
    """

    def __init__(self, Kp, Ki, Kd):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)

        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None

        self.u_unsat = None
        self.u_sat = None

        self._clear_preview_state()

    def _clear_preview_state(self):
        """
        Clear all temporary preview-state variables.

        This is called during initialization, reset(), and after commit().
        """
        self._preview_valid = False
        self._candidate_integral = None
        self._candidate_error = None
        self._candidate_derivative = None
        self._candidate_measurement = None
        self._candidate_u_unsat = None
        self._candidate_u_sat = None
        self._candidate_internal_allow_integrate = None

    @staticmethod
    def _validate_limits(u_min, u_max):
        """
        Validate optional output limits.

        Parameters
        ----------
        u_min : float or None
            Lower saturation bound.
        u_max : float or None
            Upper saturation bound.

        Raises
        ------
        ValueError
            If exactly one bound is provided or if u_min > u_max.
        """
        if (u_min is None) != (u_max is None):
            raise ValueError("u_min and u_max must either both be provided or both be None.")

        if u_min is not None:
            u_min = float(u_min)
            u_max = float(u_max)
            if u_min > u_max:
                raise ValueError(
                    f"u_min must be <= u_max. Got u_min={u_min}, u_max={u_max}."
                )

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

        self._clear_preview_state()

    def _compute_step(
        self,
        target,
        measurement,
        dt,
        u_min=None,
        u_max=None,
        u_bias=0.0,
        internal_anti_windup=True,
    ):
        """
        Compute one candidate PID step without committing controller state.

        Parameters
        ----------
        target : float
            Desired setpoint.
        measurement : float
            Current measured process value.
        dt : float
            Timestep in seconds. Must be positive.
        u_min : float or None, optional
            Minimum allowed controller output. If None, no lower saturation
            is applied.
        u_max : float or None, optional
            Maximum allowed controller output. If None, no upper saturation
            is applied.
        u_bias : float, optional
            Nominal control value about which the PID makes corrections.
            Default is 0.0.
        internal_anti_windup : bool, optional
            If True, compute the original PID-level conditional-integration
            anti-windup decision based on controller output saturation.
            Default is True.

        Returns
        -------
        dict
            Dictionary containing the full candidate step:
            - error
            - derivative
            - integral_candidate
            - u_unsat
            - u_sat
            - internal_allow_integrate

        Raises
        ------
        ValueError
            If dt <= 0 or the limits are invalid.
        """
        dt = float(dt)
        target = float(target)
        measurement = float(measurement)
        u_bias = float(u_bias)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got dt={dt}.")

        self._validate_limits(u_min, u_max)

        if u_min is not None:
            u_min = float(u_min)
            u_max = float(u_max)

        error = target - measurement

        # Derivative on measurement reduces derivative kick from sudden setpoint changes.
        if self.prev_measurement is None:
            d_meas_dt = 0.0
        else:
            d_meas_dt = (measurement - self.prev_measurement) / dt

        # Candidate integral state for this timestep.
        integral_candidate = self.integral + error * dt

        # Raw controller output before any output limiting.
        u_unsat = (
            u_bias
            + self.Kp * error
            + self.Ki * integral_candidate
            - self.Kd * d_meas_dt
        )

        # Saturated controller output, if bounds are provided.
        if u_min is None:
            u_sat = u_unsat
        else:
            u_sat = saturation(u_unsat, u_min, u_max)

        # Original internal PID anti-windup logic. This is useful for plain SISO
        # use, but may be disabled when outer-loop actuator-based anti-windup is used.
        if internal_anti_windup and (u_min is not None):
            pushing_further_high = (u_unsat > u_max) and (error > 0.0)
            pushing_further_low = (u_unsat < u_min) and (error < 0.0)
            internal_allow_integrate = not (pushing_further_high or pushing_further_low)
        else:
            internal_allow_integrate = True

        return {
            "error": error,
            "derivative": d_meas_dt,
            "integral_candidate": integral_candidate,
            "u_unsat": u_unsat,
            "u_sat": u_sat,
            "internal_allow_integrate": internal_allow_integrate,
            "measurement": measurement,
        }

    def preview(
        self,
        target,
        measurement,
        dt,
        u_min=None,
        u_max=None,
        u_bias=0.0,
        internal_anti_windup=True,
    ):
        """
        Compute a candidate PID step without committing controller state.

        This method is intended for advanced control flows, such as MIMO control
        with actuator-based anti-windup. It computes and stores the candidate
        controller state internally, then returns the candidate output and terms.
        The caller must later call commit() to finalize the step.

        Parameters
        ----------
        target : float
            Desired setpoint.
        measurement : float
            Current measured process value.
        dt : float
            Timestep in seconds. Must be positive.
        u_min : float or None, optional
            Minimum allowed controller output. If None, no lower saturation
            is applied.
        u_max : float or None, optional
            Maximum allowed controller output. If None, no upper saturation
            is applied.
        u_bias : float, optional
            Nominal control value about which the PID makes corrections.
            Default is 0.0.
        internal_anti_windup : bool, optional
            If True, compute the original PID-level conditional-integration
            anti-windup decision based on controller output saturation.
            Default is True.

        Returns
        -------
        tuple
            A 4-tuple:

            (u_sat, error, integral_candidate, derivative)

            where:
            - u_sat is the candidate controller output after saturation
            - error is the candidate control error
            - integral_candidate is the uncommitted integral state
            - derivative is d(measurement)/dt

        Notes
        -----
        preview() does not modify:
        - self.integral
        - self.prev_measurement
        - self.error
        - self.derivative
        - self.u_unsat
        - self.u_sat

        Those states are only updated when commit() is called.
        """
        result = self._compute_step(
            target=target,
            measurement=measurement,
            dt=dt,
            u_min=u_min,
            u_max=u_max,
            u_bias=u_bias,
            internal_anti_windup=internal_anti_windup,
        )

        self._candidate_error = result["error"]
        self._candidate_derivative = result["derivative"]
        self._candidate_integral = result["integral_candidate"]
        self._candidate_measurement = result["measurement"]
        self._candidate_u_unsat = result["u_unsat"]
        self._candidate_u_sat = result["u_sat"]
        self._candidate_internal_allow_integrate = result["internal_allow_integrate"]
        self._preview_valid = True

        return (
            self._candidate_u_sat,
            self._candidate_error,
            self._candidate_integral,
            self._candidate_derivative,
        )

    def commit(self, allow_integrate=True):
        """
        Finalize the most recent preview() step.

        Parameters
        ----------
        allow_integrate : bool, optional
            If True, allow the candidate integral state to be committed,
            subject to the internal PID anti-windup decision computed during
            preview(). If False, the integral is frozen for this step.
            Default is True.

        Returns
        -------
        tuple
            A 4-tuple:

            (u_sat, error, integral, derivative)

            where:
            - u_sat is the committed controller output after saturation
            - error is the committed control error
            - integral is the committed integral state
            - derivative is the committed derivative term

        Raises
        ------
        RuntimeError
            If commit() is called before preview().

        Notes
        -----
        The final integral commit decision is:

            final_allow_integrate =
                allow_integrate AND internal_allow_integrate

        This lets the outer controller enforce actuator-based anti-windup while
        still optionally preserving the original PID-level anti-windup.
        """
        if not self._preview_valid:
            raise RuntimeError("commit() called without a valid preview().")

        final_allow_integrate = bool(allow_integrate) and bool(
            self._candidate_internal_allow_integrate
        )

        self.error = self._candidate_error
        self.derivative = self._candidate_derivative
        self.u_unsat = self._candidate_u_unsat
        self.u_sat = self._candidate_u_sat

        if final_allow_integrate:
            self.integral = self._candidate_integral

        # Even when the integral is frozen, time has advanced and a new
        # measurement has been observed, so the derivative state should update.
        self.prev_measurement = self._candidate_measurement

        out = (self.u_sat, self.error, self.integral, self.derivative)

        self._clear_preview_state()

        return out

    def update(
        self,
        target,
        measurement,
        dt,
        u_min=None,
        u_max=None,
        u_bias=0.0,
        internal_anti_windup=True,
    ):
        """
        Advance the PID controller by one timestep and immediately commit it.

        This is a convenience wrapper around:

            preview(...)
            commit(allow_integrate=True)

        Parameters
        ----------
        target : float
            Desired setpoint.
        measurement : float
            Current measured process value.
        dt : float
            Timestep in seconds. Must be positive.
        u_min : float or None, optional
            Minimum allowed controller output. If None, no lower saturation
            is applied.
        u_max : float or None, optional
            Maximum allowed controller output. If None, no upper saturation
            is applied.
        u_bias : float, optional
            Nominal control value about which the PID makes corrections.
            Default is 0.0.
        internal_anti_windup : bool, optional
            If True, apply the original PID-level conditional-integration
            anti-windup against PID output saturation.
            Default is True.

        Returns
        -------
        tuple
            A 4-tuple:

            (u_sat, error, integral, derivative)

            where:
            - u_sat is the committed controller output after saturation
            - error is the current control error
            - integral is the committed integral state
            - derivative is d(measurement)/dt

        Notes
        -----
        For standard SISO use, this behaves similarly to the original PID
        interface, while allowing per-call limits and bias.

        For outer-loop actuator-based anti-windup, use preview() and commit()
        instead so the outer controller can decide whether integration should
        be allowed after checking actuator saturation or rate limiting.
        """
        self.preview(
            target=target,
            measurement=measurement,
            dt=dt,
            u_min=u_min,
            u_max=u_max,
            u_bias=u_bias,
            internal_anti_windup=internal_anti_windup,
        )
        return self.commit(allow_integrate=True)

    def __repr__(self):
        """
        Return a developer-friendly representation of the controller state.
        """
        return (
            f"{self.__class__.__name__}("
            f"Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
            f"error={self.error}, integral={self.integral}, derivative={self.derivative}, "
            f"u_unsat={self.u_unsat}, u_sat={self.u_sat})"
        )
        
'''