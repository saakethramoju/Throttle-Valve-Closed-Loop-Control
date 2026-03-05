from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import root


def solve_system_CdAs(
    ts,
    *,
    Pc_target: float,
    MR_target: float,
    x0_CdA: tuple[float, float] | None = None,
    tol: float = 1e-9,
    maxfev: int = 250,
    CdA_min: float = 1e-10,
    CdA_max: float = 1e-1,
    Pc_scale: float | None = None,
    MR_scale: float | None = None,
    verbose: bool = False,
):
    """
    Solve for system throttle CdAs that achieve a target chamber pressure and mixture ratio.

    This routine determines the effective discharge areas of the fuel and oxidizer
    **system throttles** such that the steady-state solution of the TestStand satisfies:

        Pc(ts) ≈ Pc_target
        MR(ts) ≈ MR_target

    where Pc is the combustion chamber pressure and MR is the oxidizer-to-fuel
    mass mixture ratio.

    The solution is obtained by wrapping the TestStand steady-state solver
    inside a nonlinear root-finding problem using SciPy's `root(method="hybr")`.

    The unknowns are:

        CdA_fuel_system
        CdA_oxidizer_system

    which correspond to:

        ts.FuelThrottleValve.CdA
        ts.OxThrottleValve.CdA


    --------------------------------------------------------------------------
    Solver Strategy
    --------------------------------------------------------------------------

    The root solver operates on a transformed variable:

        u = log(CdA)

    so that:

        CdA = exp(u)

    This guarantees that CdA remains strictly positive during iteration,
    preventing the solver from exploring non-physical negative flow areas.

    During each residual evaluation:

    1. The CdAs are reconstructed from the log variables.
    2. They are clamped to `[CdA_min, CdA_max]` to avoid pathological values.
    3. The throttle CdAs of a working copy of the TestStand are updated.
    4. The TestStand steady-state solver is executed.
    5. Residuals are computed as normalized errors:

            r1 = (Pc - Pc_target) / Pc_scale
            r2 = (MR - MR_target) / MR_scale

    These residuals are returned to the root solver.


    --------------------------------------------------------------------------
    Warm-Starting the Inner Solver
    --------------------------------------------------------------------------

    The steady-state solver itself performs a nonlinear solve for internal
    pressures (injector manifold pressures and chamber pressure).

    To accelerate convergence, this routine **warm-starts** each new
    steady-state solve using the previous successful solution:

        last_x0 = solved.solved_x0()

    This dramatically reduces the number of inner solver iterations when
    the outer CdA iteration makes only small changes.


    --------------------------------------------------------------------------
    Copy Semantics
    --------------------------------------------------------------------------

    The original TestStand object provided by the caller is **never mutated**.

    Instead:

        base = deepcopy(ts)

    is created once at the start of the solve. Only this internal working copy
    has its CdAs modified during the iteration.

    After convergence, a **fresh deep copy of the caller's TestStand** is
    created and the solved CdAs are applied before running one final
    steady-state solve to produce the returned result.

    A custom `TestStand.__deepcopy__` implementation ensures that the
    RocketCEA object (`_cea_obj`) is shared rather than copied, avoiding
    expensive duplication of thermochemistry objects.


    --------------------------------------------------------------------------
    Failure Handling
    --------------------------------------------------------------------------

    If the steady-state solver fails at a candidate CdA pair, the routine
    does **not raise immediately**. Instead it returns a large penalty
    residual to the root solver so that iteration may continue.

    This prevents occasional infeasible points from terminating the solve.


    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    ts : TestStand
        Baseline test stand configuration. This object is treated as
        immutable and will not be modified by the solver.

    Pc_target : float
        Desired combustion chamber pressure [Pa].

    MR_target : float
        Desired oxidizer-to-fuel mass mixture ratio.

    x0_CdA : tuple[float, float] | None, optional
        Initial guess for the system CdAs:

            (FuelThrottleValve.CdA, OxThrottleValve.CdA)

        If None, the values currently stored in the TestStand are used.

    tol : float, optional
        Root-finding convergence tolerance passed to SciPy's `root`
        (`xtol` parameter).

    maxfev : int, optional
        Maximum number of residual evaluations allowed by the root solver.

    CdA_min : float, optional
        Lower bound applied to CdA during iteration to prevent degenerate
        flow elements.

    CdA_max : float, optional
        Upper bound applied to CdA to prevent unrealistically large flow areas.

    Pc_scale : float | None, optional
        Scaling factor used when forming the chamber pressure residual.

        If None, it defaults to:

            max(abs(Pc_target), 1)

        which keeps the residual magnitude O(1).

    MR_scale : float | None, optional
        Scaling factor used when forming the mixture ratio residual.

    verbose : bool, optional
        If True, prints iteration diagnostics including CdA guesses,
        resulting chamber pressure, mixture ratio, and residual values.


    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    tuple
        A tuple containing:

        CdA_fuel : float
            Solved fuel throttle CdA [m²]

        CdA_ox : float
            Solved oxidizer throttle CdA [m²]

        solved_teststand : TestStand
            A fully solved TestStand instance containing the converged
            steady-state operating point.


    --------------------------------------------------------------------------
    Raises
    --------------------------------------------------------------------------
    ValueError
        If the initial CdA guess is non-positive.

    RuntimeError
        If the root solver fails to converge.


    --------------------------------------------------------------------------
    Notes
    --------------------------------------------------------------------------

    • This routine is most useful when designing feed systems where the
      system throttles must be sized to achieve a desired operating point.

    • Because each residual evaluation requires a steady-state solve, the
      computational cost is dominated by the inner TestStand solver.

    • The global RocketCEA object cache and the custom TestStand deepcopy
      implementation ensure that thermochemistry objects are **not recreated
      or copied during the solve**, keeping the iteration efficient.

    """

    base = copy.deepcopy(ts)  # safe sandbox (won't recreate CEA if you cache + __deepcopy__)
    last_x0 = base.get_x0()

    if x0_CdA is None:
        x0_CdA = (float(base.FuelThrottleValve.CdA), float(base.OxThrottleValve.CdA))

    CdA_f0, CdA_ox0 = map(float, x0_CdA)
    if CdA_f0 <= 0 or CdA_ox0 <= 0:
        raise ValueError("Initial CdA guesses must be > 0.")

    # scaling to keep residuals balanced
    if Pc_scale is None:
        Pc_scale = max(abs(Pc_target), 1.0)
    if MR_scale is None:
        MR_scale = max(abs(MR_target), 1.0)

    BIG = 1e9

    def clamp(x: float) -> float:
        return float(np.clip(x, CdA_min, CdA_max))

    # log-transform => always positive
    u0 = np.log([CdA_f0, CdA_ox0])

    def residual_u(u: np.ndarray) -> np.ndarray:
        nonlocal last_x0

        CdA_f = clamp(np.exp(float(u[0])))
        CdA_ox = clamp(np.exp(float(u[1])))

        # mutate only CdAs on base
        base.FuelThrottleValve.CdA = CdA_f
        base.OxThrottleValve.CdA   = CdA_ox

        try:
            solved = base.steady_state(x0=last_x0)
            last_x0 = solved.solved_x0()

            Pc = float(solved.MainChamber.p)
            MR = float(getattr(solved.MainChamber, "MR", np.nan))
            if not np.isfinite(Pc) or not np.isfinite(MR) or Pc <= 0 or MR <= 0:
                return np.array([BIG, BIG], dtype=float)

            rPc = (Pc - Pc_target) / Pc_scale
            rMR = (MR - MR_target) / MR_scale

            if verbose:
                print(f"CdA_f={CdA_f:.3e}, CdA_ox={CdA_ox:.3e} -> Pc={Pc:.3e}, MR={MR:.4f}, r=[{rPc:.3e}, {rMR:.3e}]")

            return np.array([rPc, rMR], dtype=float)

        except Exception as e:
            if verbose:
                print(f"steady_state failed at CdA_f={CdA_f:.3e}, CdA_ox={CdA_ox:.3e}: {e}")
            return np.array([BIG, BIG], dtype=float)

    sol = root(residual_u, x0=u0, method="hybr", options={"xtol": tol, "maxfev": int(maxfev)})

    if not sol.success:
        CdA_f_best = clamp(np.exp(float(sol.x[0])))
        CdA_ox_best = clamp(np.exp(float(sol.x[1])))
        raise RuntimeError(
            "solve_system_CdAs failed.\n"
            f"message: {sol.message}\n"
            f"CdA best guess: fuel={CdA_f_best:.6e}, ox={CdA_ox_best:.6e}"
        )

    CdA_f_sol = clamp(np.exp(float(sol.x[0])))
    CdA_ox_sol = clamp(np.exp(float(sol.x[1])))

    # apply to a fresh copy of the caller and return a final solved state
    out = copy.deepcopy(ts)
    out.FuelThrottleValve.CdA = CdA_f_sol
    out.OxThrottleValve.CdA   = CdA_ox_sol
    out_solved = out.steady_state()

    return CdA_f_sol, CdA_ox_sol, out_solved