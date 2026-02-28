from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import root

from .Components import *
from Utilities import choked_nozzle_thrust, choked_nozzle_mass_flow, create_CEA_object
from Utilities import incompressible_CdA_equation
from Physics.Constants import *

from .Balance import Balance


class TestStand:
    """
    Lumped-parameter test stand model for steady-state feed + injector + chamber + nozzle coupling.

    This object composes component instances representing the fuel and oxidizer
    flow paths feeding a combustion chamber and a choked nozzle throat.

    Assumptions
    ------------------------------------
    - Incompressible flow through CdA elements for feed system + injectors.
    - No two-phase effects, flashing, cavitation, or temperature coupling.

    Parameters
    ----------
    name: str
        Identifier for test stand.
    FuelTank, OxTank : Tank
        Tanks holding propellants, providing upstream pressure and density.
        Expected attributes used here include: `.p` [Pa], `.rho` [kg/m^3], and `.propellant` (name).
    FuelRunline, OxRunline : Line
        Feed lines. s
    FuelThrottleValve, OxThrottleValve : Valve
        Throttle elements providing system CdA on each side.
        Expected attribute: `.CdA` [m^2].
    FuelInjectorManifold, OxInjectorManifold : InjectorManifold
        Manifolds upstream of injector orifices; used for initial guesses and may store solved pressures.
        Expected attribute: `.p` [Pa].
    FuelInjector, OxInjector : Orifice
        Injector elements modeled as incompressible CdA elements.
        Expected attributes: `.CdA` [m^2]; this solver will also populate `.mdot` [kg/s] if present.
    MainChamber : CombustionChamber
        Combustion chamber model; expected attributes: `.p` [Pa] initial guess and `.eta_cstar` (dimensionless).
    TCA : Nozzle
        Nozzle/throat model; expected attribute: `.At` [m^2].
    Ambient : Drain
        Ambient atmospheric model; expected attribute: `.p` [Pa].

    Notes
    -----
    - This class is designed so you can keep an "unsolved" configuration `A` and generate a solved
      configuration `B` without mutating `A`.
    
    - If, for any reason, the layout of TestStand needs to be altered (whether a new volume node needs
      to be added, or the entire system needs to be reconfigured), because this code is not entirely 
      modular, several keys items must be modified:

      1) Any hardcoded attributes references (e.g. self.FuelTank.p, self.MainChamber.MR) need to be
         modified in the class and the residual function.
    
      2) The number of residuals and iteration variables in the residual function should be altered.
         Currently, len(STEADY_STATE_X0_PATHS) == 3 is used enfore three iteration variables, and
         hence, three residuals. Make sure to alter this guarding code.

      3) STEADY_STATE_X0_PATHS must be updated. The order of the list matters! STEADY_STATE_X0_PATHS
         tells the solver how many variables are iterated on. The order corresponds to the iteration
         variables in the residual function. STEADY_STATE_X0_PATHS can be found above __init__().

      4) __str__ and __repr__ are configuration dependent.

    - Balance.py should agnostic to layout changes, unless the way that component attributes are 
      accessed changes (e.g. FuelSide.Tank.p instead of FuelTank.p)

    - steady_state_with_balance() should also continue to be generic, working for most any layout changes,
      as long as there is a steady_state() function that accepts x0, a get_x0(), and a solved_x0().
      Make sure to also update the residual function and STEADY_STATE_X0_PATHS, and everything should
      be fine.
        
    """


    # Order matters: MUST match the unknown ordering used in residual(x)
    STEADY_STATE_X0_PATHS = [
        "FuelInjectorManifold.p",
        "OxInjectorManifold.p",
        "MainChamber.p",
    ]

    def __init__(self, name: str,
                 FuelTank: Tank, OxTank: Tank,
                 FuelRunline: Line, OxRunline: Line,
                 FuelThrottleValve: Valve, OxThrottleValve: Valve,
                 FuelInjectorManifold: InjectorManifold, OxInjectorManifold: InjectorManifold,
                 FuelInjector: Orifice, OxInjector: Orifice,
                 MainChamber: CombustionChamber, TCA: Nozzle,
                 Ambient: Drain):

        self.name = name
        self.FuelTank = FuelTank
        self.OxTank = OxTank
        self.FuelRunline = FuelRunline
        self.OxRunline = OxRunline
        self.FuelThrottleValve = FuelThrottleValve
        self.OxThrottleValve = OxThrottleValve
        self.FuelInjectorManifold = FuelInjectorManifold
        self.OxInjectorManifold = OxInjectorManifold
        self.FuelInjector = FuelInjector
        self.OxInjector = OxInjector
        self.MainChamber = MainChamber
        self.TCA = TCA
        self.Ambient = Ambient

        fuel_name = self.FuelTank.propellant
        ox_name   = self.OxTank.propellant
        self._cea_obj = create_CEA_object(fuel_name, ox_name)

    def __repr__(self) -> str:
        return f"TestStand (name={self.name}, FuelTank={self.FuelTank!r}, OxTank={self.OxTank!r}, MainChamber={self.MainChamber!r}, TCA={self.TCA!r})"


    def __str__(self, units: str = "SI") -> str:
        # Local helpers
        def _fmt_kgps(x):
            return f"{x:.4f} kg/s" if x is not None else "—"

        def _get(obj, attr):
            v = getattr(obj, attr, None)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        def _table(rows, headers):
            """
            rows: list of tuples/iterables of strings
            headers: tuple/list of strings
            """
            cols = list(zip(*([headers] + rows)))
            widths = [max(len(str(c)) for c in col) for col in cols]

            def fmt_row(r):
                return " | ".join(str(c).ljust(w) for c, w in zip(r, widths))

            sep = "-+-".join("-" * w for w in widths)
            out = [fmt_row(headers), sep]
            out += [fmt_row(r) for r in rows]
            return "\n".join(out)

        # ------------------------
        # Unit system selection
        # ------------------------
        units = (units or "SI").upper()

        if units == "SI":
            def _fmt_pressure(p):
                return f"{p:,.2f} Pa" if p is not None else "—"

            def _fmt_area(a):
                return f"{a:.3e} m^2" if a is not None else "—"

            def _fmt_force(f):
                return f"{f:,.2f} N" if f is not None else "—"

        elif units == "US":
            def _fmt_pressure(p):
                return f"{p / PA_PER_PSI:,.2f} psi" if p is not None else "—"

            def _fmt_area(a):
                return f"{a * IN2_PER_M2:.4f} in^2" if a is not None else "—"

            def _fmt_force(f):
                return f"{f * LBF_PER_N:,.2f} lbf" if f is not None else "—"

        else:
            raise ValueError("units must be 'SI' or 'US'")

        # Pull common numbers (don’t crash if unsolved)
        mdot_f = _get(self.FuelInjector, "mdot")
        mdot_ox = _get(self.OxInjector, "mdot")
        mdot_total = (mdot_f + mdot_ox) if (mdot_f is not None and mdot_ox is not None) else None
        MR = (mdot_ox / mdot_f) if (mdot_f is not None and mdot_f != 0 and mdot_ox is not None) else None

        # Pressures
        P_tank_f = _get(self.FuelTank, "p")
        P_tank_ox = _get(self.OxTank, "p")
        P_inj_f = _get(self.FuelInjectorManifold, "p")
        P_inj_ox = _get(self.OxInjectorManifold, "p")
        Pc = _get(self.MainChamber, "p")
        Pamb = _get(self.Ambient, "p")

        # Densities (if you store them)
        rho_f = _get(self.FuelTank, "rho")
        rho_ox = _get(self.OxTank, "rho")

        rho_f_man = _get(self.FuelInjectorManifold, "rho")
        rho_ox_man = _get(self.OxInjectorManifold, "rho")

        # Nozzle / thrust
        At = _get(self.TCA, "At")
        eps = _get(self.TCA, "eps")
        F = _get(self.TCA, "F")

        # CdAs (if present)
        CdA_f_sys = _get(self.FuelThrottleValve, "CdA")
        CdA_ox_sys = _get(self.OxThrottleValve, "CdA")
        CdA_f_inj = _get(self.FuelInjector, "CdA")
        CdA_ox_inj = _get(self.OxInjector, "CdA")

        # Injector stiffness (dimensionless dp/p)
        stiff_f = _get(self.FuelInjector, "stiffness")
        stiff_ox = _get(self.OxInjector, "stiffness")

        # Build tables
        state_rows = [
            ("Fuel Tank",        _fmt_pressure(P_tank_f),   f"{rho_f:.1f} kg/m^3" if rho_f is not None else "—"),
            ("Fuel Manifold",    _fmt_pressure(P_inj_f),    f"{rho_f_man:.1f} kg/m^3" if rho_f_man is not None else "—"),
            ("Ox Tank",          _fmt_pressure(P_tank_ox),  f"{rho_ox:.1f} kg/m^3" if rho_ox is not None else "—"),
            ("Ox Manifold",      _fmt_pressure(P_inj_ox),   f"{rho_ox_man:.1f} kg/m^3" if rho_ox_man is not None else "—"),
            ("Chamber",          _fmt_pressure(Pc),         "—"),
            ("Ambient",          _fmt_pressure(Pamb),       "—"),
        ]

        flow_rows = [
            ("Fuel mdot",  _fmt_kgps(mdot_f)),
            ("Ox mdot",    _fmt_kgps(mdot_ox)),
            ("Total mdot", _fmt_kgps(mdot_total)),
            ("MR",   f"{MR:.4f}" if MR is not None else "—"),
            ("Fuel inj stiffness", f"{100.0 * stiff_f:.2f} %" if stiff_f is not None else "—"),
            ("Ox inj stiffness",   f"{100.0 * stiff_ox:.2f} %" if stiff_ox is not None else "—"),
        ]

        geom_rows = [
            ("At",  _fmt_area(At)),
            ("eps", f"{eps:.3f}" if eps is not None else "—"),
            ("F",   _fmt_force(F)),
        ]

        cda_rows = [
            ("Fuel throttle CdA", _fmt_area(CdA_f_sys)),
            ("Ox throttle CdA",   _fmt_area(CdA_ox_sys)),
            ("Fuel injector CdA", _fmt_area(CdA_f_inj)),
            ("Ox injector CdA",   _fmt_area(CdA_ox_inj)),
        ]

        # Final report string
        return (
            f"\n================ {self.name} =================\n"
            f"\n[Pressures / Densities]\n{_table(state_rows, headers=('Node', 'Pressure', 'Density'))}\n"
            f"\n[Mass Flow]\n{_table(flow_rows, headers=('Quantity', 'Value'))}\n"
            f"\n[Nozzle / Performance]\n{_table(geom_rows, headers=('Parameter', 'Value'))}\n"
            f"\n[CdA Summary]\n{_table(cda_rows, headers=('Element', 'Value'))}\n"
            f"================================================\n"
        )


    @classmethod
    def _get_attr_by_path(cls, obj, path: str) -> float:
        """
        Resolve a dotted path like 'FuelInjectorManifold.p' on obj.
        Only supports 2-level paths to match your Balance style.
        """
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'Object.attr' path, got {path!r}")
        comp, attr = parts
        return float(getattr(getattr(obj, comp), attr))


    @classmethod
    def _set_attr_by_path(cls, obj, path: str, value: float) -> None:
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'Object.attr' path, got {path!r}")
        comp, attr = parts
        setattr(getattr(obj, comp), attr, float(value))


    def get_x0(self) -> list[float]:
        """
        Default initial guess for steady_state(), based on STEADY_STATE_X0_PATHS.
        """
        return [self._get_attr_by_path(self, p) for p in self.STEADY_STATE_X0_PATHS]


    def solved_x0(self) -> list[float]:
        """
        Warm-start x0 extracted from a SOLVED TestStand.
        Uses the same STEADY_STATE_X0_PATHS contract.
        """
        return [self._get_attr_by_path(self, p) for p in self.STEADY_STATE_X0_PATHS]
    


    def steady_state(
        self,
        x0: list[float] | None = None,
        tol: float = 1e-9,
        maxfev: int | None = None,
    ) -> "TestStand":
        """
        Solve for steady-state injector-manifold pressures and chamber pressure.

        Solves the nonlinear system for:
        - P_inj_f  : fuel injector upstream (manifold) pressure [Pa]
        - P_inj_ox : oxidizer injector upstream (manifold) pressure [Pa]
        - Pc       : chamber pressure [Pa]

        The residual enforces:
        1) Fuel continuity:
                mdot_sys_fuel(P_tank_f -> P_inj_f) = mdot_inj_fuel(P_inj_f -> Pc)
        2) Ox continuity:
                mdot_sys_ox(P_tank_ox -> P_inj_ox) = mdot_inj_ox(P_inj_ox -> Pc)
        3) Chamber/nozzle continuity:
                mdot_inj_fuel + mdot_inj_ox = mdot_nozzle_choked(Pc, MR, At, eta_cstar)

        After convergence, this method returns a NEW `TestStand` instance (deep-copied)
        with solved pressures and computed mass flows written into the relevant components.
        If ambient pressure is available (via `self.Ambient.p`), it also
        computes nozzle thrust using RocketCEA Cf.

        Parameters
        ----------
        x0 : list[float] or None, optional
            Initial guess [P_inj_f, P_inj_ox, Pc] in Pa.
            If None, uses [FuelInjectorManifold.p, OxInjectorManifold.p, MainChamber.p].
        tol : float, optional
            Root-finder tolerance for SciPy `root(method="hybr")`. Default 1e-9.
        maxfev : int or None, optional
            Maximum function evaluations for the solver. If None, SciPy default.

        Returns
        -------
        TestStand
            A NEW TestStand instance containing solved pressures, mass flow rates,
            mixture ratio, and (optionally) thrust. The original object is not mutated.

        Raises
        ------
        ValueError
            If required physical inputs are non-positive (areas, densities, CdAs, etc.).
        RuntimeError
            If SciPy `root` fails to converge.

        Notes
        -----
        - Feed and injector flows are modeled as incompressible CdA elements.
        - Nozzle mass flow is assumed choked and computed via c* (RocketCEA) and eta_cstar.
        - Ensure Pc/Pamb units match what your RocketCEA wrapper expects.
        """

        # --- Basic validation ---

        if len(self.STEADY_STATE_X0_PATHS) != 3:
            raise ValueError(
                "This steady_state() implementation currently solves exactly 3 unknowns "
                "(FuelInjectorManifold.p, OxInjectorManifold.p, MainChamber.p). "
                "This likely means the TestStand layout has been altered to include more volume nodes. "
                "The number of residuals must now be updated. "
                "Update residual(x) if you change STEADY_STATE_X0_PATHS."
            )
        
        if self.TCA.At <= 0:
            raise ValueError("Nozzle throat area At must be > 0.")
        if self.MainChamber.eta_cstar <= 0:
            raise ValueError("MainChamber.eta_cstar must be > 0.")
        if self.FuelTank.p <= 0 or self.OxTank.p <= 0:
            raise ValueError("Tank pressures must be > 0 Pa.")
        if self.FuelTank.rho <= 0 or self.OxTank.rho <= 0:
            raise ValueError("Tank densities must be > 0 kg/m^3.")
        if self.FuelThrottleValve.CdA <= 0 or self.OxThrottleValve.CdA <= 0:
            raise ValueError("Throttle valve CdA values must be > 0.")
        if self.FuelInjector.CdA <= 0 or self.OxInjector.CdA <= 0:
            raise ValueError("Injector CdA values must be > 0.")

        fuel_name = self.FuelTank.propellant
        ox_name = self.OxTank.propellant
        cea_obj = self._cea_obj

        P_tank_f = float(self.FuelTank.p)
        P_tank_ox = float(self.OxTank.p)
        rho_f = float(self.FuelTank.rho)
        rho_ox = float(self.OxTank.rho)

        sys_CdA_f = float(self.FuelThrottleValve.CdA)
        sys_CdA_ox = float(self.OxThrottleValve.CdA)
        inj_CdA_f = float(self.FuelInjector.CdA)
        inj_CdA_ox = float(self.OxInjector.CdA)

        At = float(self.TCA.At)
        eta_cstar = float(self.MainChamber.eta_cstar)

        def residual(x: np.ndarray) -> np.ndarray:
            P_inj_f, P_inj_ox, Pc = map(float, x)

            # Keep solver away from non-physical space
            if P_inj_f <= 0 or P_inj_ox <= 0 or Pc <= 0:
                return np.array([1e12, 1e12, 1e12], dtype=float)

            sys_fuel_mdot = incompressible_CdA_equation(P_tank_f, P_inj_f, rho_f, sys_CdA_f)
            sys_ox_mdot = incompressible_CdA_equation(P_tank_ox, P_inj_ox, rho_ox, sys_CdA_ox)

            inj_fuel_mdot = incompressible_CdA_equation(P_inj_f, Pc, rho_f, inj_CdA_f)
            inj_ox_mdot = incompressible_CdA_equation(P_inj_ox, Pc, rho_ox, inj_CdA_ox)

            # Guard MR
            # Reject non-physical injector flows before calling CEA
            if inj_fuel_mdot <= 0 or inj_ox_mdot <= 0:
                return np.array([1e12, 1e12, 1e12], dtype=float)

            MR = inj_ox_mdot / inj_fuel_mdot

            # Also keep MR in a sane band for CEA 
            if not (0.5 <= MR <= 6.0):
                return np.array([1e12, 1e12, 1e12], dtype=float)

            # Use the correct function name in your codebase:
            # If your helper is named choked_mass_flow, use that.
            tca_mdot = choked_nozzle_mass_flow(Pc, MR, At, eta_cstar, cea_obj)

            return np.array([
                sys_fuel_mdot - inj_fuel_mdot,
                sys_ox_mdot - inj_ox_mdot,
                (inj_fuel_mdot + inj_ox_mdot) - tca_mdot
            ], dtype=float)

        if x0 is None:
            x0 = self.get_x0()

        expected = len(self.STEADY_STATE_X0_PATHS)
        if len(x0) != expected:
            raise ValueError(
                f"steady_state expected x0 of length {expected} "
                f"(matching STEADY_STATE_X0_PATHS), got {len(x0)}."
            )

        options = {"xtol": tol}
        if maxfev is not None:
            options["maxfev"] = int(maxfev)

        sol = root(residual, x0=np.array(x0, dtype=float), method="hybr", options=options)

        if not sol.success:
            raise RuntimeError(
                "Steady-state solve failed.\n"
                f"message: {sol.message}\n"
                f"x0     : {x0}\n"
                f"x      : {sol.x}"
            )

        P_inj_f_sol, P_inj_ox_sol, Pc_sol = map(float, sol.x)

        solved = copy.deepcopy(self)

        # Compute final injector mdots at solved pressures
        solved_fuel_mdot = incompressible_CdA_equation(P_inj_f_sol, Pc_sol, rho_f, inj_CdA_f)
        solved_ox_mdot = incompressible_CdA_equation(P_inj_ox_sol, Pc_sol, rho_ox, inj_CdA_ox)

        # Write back mdots
        solved.FuelRunline.mdot = solved_fuel_mdot
        solved.OxRunline.mdot = solved_ox_mdot

        solved.FuelThrottleValve.mdot = solved_fuel_mdot
        solved.OxThrottleValve.mdot = solved_ox_mdot

        # Write back manifold pressures + densities
        solved.FuelInjectorManifold.p = P_inj_f_sol
        if hasattr(solved.FuelInjectorManifold, "rho"):
            solved.FuelInjectorManifold.rho = rho_f

        solved.OxInjectorManifold.p = P_inj_ox_sol
        if hasattr(solved.OxInjectorManifold, "rho"):
            solved.OxInjectorManifold.rho = rho_ox  

        # Write back injector mdots
        solved.FuelInjector.mdot = solved_fuel_mdot
        solved.OxInjector.mdot = solved_ox_mdot

        # Chamber state
        solved.MainChamber.p = Pc_sol
        MR_sol = (solved_ox_mdot / solved_fuel_mdot) if solved_fuel_mdot > 0 else float("inf")
        if hasattr(solved.MainChamber, "MR"):
            solved.MainChamber.MR = MR_sol

        solved.FuelInjector.stiffness = (P_inj_f_sol - Pc_sol) / Pc_sol
        solved.OxInjector.stiffness = (P_inj_ox_sol - Pc_sol) / Pc_sol

        # Nozzle state
        solved.TCA.mdot = solved_fuel_mdot + solved_ox_mdot

        solved.TCA.F = choked_nozzle_thrust(
            Pc_sol, MR_sol, solved.TCA.At, self.Ambient.p,
            solved.TCA.eps, solved.TCA.eta_cf, solved.TCA.nfz, cea_obj
        )

        return solved
    


    def steady_state_with_balance(
        self,
        balance: Balance,
        *,
        x0: list[float] | None = None,
        max_iter: int = 60,
        bracket_expand: int = 10,
        fail_penalty: float | None = None,   # kept for compatibility; currently unused
        bracket_samples: int = 8,
        verbose: bool = False
    ) -> "TestStand":
        """
        Solve a 1D "balance" problem by tuning a single scalar knob until a target
        output is met, using repeated steady-state solves and bisection.

        Conceptually, this finds a knob value k such that:

            measure_fn( steady_state(TestStand with knob=k) ) ≈ target

        where the Balance object defines:
        - what attribute to tune (balance.tune_set)
        - what quantity to measure (balance.measure_fn)
        - the target value (balance.target)
        - the allowable knob search bounds (balance.bounds)
        - the convergence tolerance on the measurement error (balance.tol)

        Method Overview
        ---------------
        1) Bracketing:
        Attempts to find two knob values [a, b] within bounds such that the signed
        error changes sign:

            err(k) = measure_fn(solved_ts(k)) - target
            err(a) * err(b) <= 0

        - First tries the endpoints (lo, hi).
        - If that fails, scans `bracket_samples` evenly spaced points in [lo, hi]
            and looks for a sign change between solvable points.
        - If bracketing still fails, optionally expands the bounds outward up to
            `bracket_expand` times (lo *= 0.5, hi *= 2.0) and retries.

        2) Bisection:
        With a valid bracket [a, b], repeatedly bisects:
            mid = (a + b)/2
        evaluates err(mid), and keeps the half-interval that preserves the sign
        change. Stops when |err(mid)| < balance.tol or after `max_iter`.

        Robustness Features
        -------------------
        - Warm-starting:
        Each successful solve updates `last_x0` using `solved.solved_x0()`.
        This makes subsequent steady_state() calls converge faster and more reliably.
        - Handling unsolvable midpoints:
        If steady_state fails at the midpoint, the algorithm nudges the evaluation
        point toward a solvable side (mid->(mid+a)/2, then mid->(mid+b)/2).

        Parameters
        ----------
        balance : Balance
            Defines the knob to tune, measurement to evaluate, target value,
            knob bounds, and tolerance.

        x0 : list[float] | None, keyword-only
            Initial guess passed into steady_state() on the first evaluation.
            If None, uses `self.get_x0()`.

        max_iter : int, keyword-only
            Maximum number of bisection iterations after a bracket is found.

        bracket_expand : int, keyword-only
            Number of times the [lo, hi] bounds may be expanded outward if a valid
            sign-changing bracket cannot be found initially.

        fail_penalty : float | None, keyword-only
            Currently unused (kept for compatibility). If you later decide to treat
            steady_state failures as a large signed error instead of skipping them,
            this value could be used as the magnitude of that penalty.

        bracket_samples : int, keyword-only
            Number of evenly spaced sample points inside [lo, hi] used to find a
            sign-changing bracket when endpoints are not sufficient.

        verbose : bool, keyword-only
            If True, prints diagnostic information when steady_state fails at a knob.

        Returns
        -------
        TestStand
            A NEW solved TestStand instance corresponding to the knob value that
            satisfies the target within tolerance (or best-effort if max_iter reached).

        Raises
        ------
        ValueError
            If the balance bounds are invalid.

        RuntimeError
            If no solvable bracketing points can be found, if the target cannot be
            bracketed (no sign change), or if repeated midpoint evaluations fail.

        Notes
        -----
        - This routine assumes the measured output changes "reasonably" with the knob.
        Bisection requires that the solution be bracketable (i.e., the error crosses 0).
        - The original TestStand object is not mutated; all knob evaluations operate
        on deep-copied versions of `self`.
        """
        lo, hi = map(float, balance.bounds)
        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError("balance.bounds must be (lo, hi) with 0 < lo < hi.")

        base = copy.deepcopy(self)
        tol_y = float(balance.tol)

        # Warm-start storage
        last_x0: list[float] | None = copy.deepcopy(x0) if x0 is not None else base.get_x0()

        def try_err_at(knob: float) -> tuple[bool, float, "TestStand" | None]:
            """
            Evaluate the signed error at a given knob value.

            Returns (ok, err, solved_ts):
            ok=False if steady_state fails at this knob.
            """
            nonlocal last_x0
            ts = copy.deepcopy(base)
            balance.tune_set(ts, float(knob))
            try:
                solved = ts.steady_state(x0=last_x0)
                last_x0 = solved.solved_x0()  # warm-start hook
                y = float(balance.measure_fn(solved))
                return True, (y - float(balance.target)), solved
            except Exception as e:
                if verbose:
                    print(f"[solve_with_balance] steady_state failed at knob={knob:.3e}: {e}")
                return False, float("nan"), None

        def find_bracket(lo: float, hi: float) -> tuple[float, float, float, float]:
            """
            Find (a, b, f_a, f_b) with solvable endpoints and f_a * f_b <= 0.
            """
            ok_lo, f_lo, _ = try_err_at(lo)
            ok_hi, f_hi, _ = try_err_at(hi)

            if ok_lo and ok_hi and (f_lo * f_hi <= 0):
                return lo, hi, f_lo, f_hi

            xs = np.linspace(lo, hi, bracket_samples)
            vals: list[tuple[float, float]] = []
            for x in xs:
                ok, fx, _ = try_err_at(float(x))
                if ok and np.isfinite(fx):
                    vals.append((float(x), float(fx)))

            if len(vals) < 2:
                raise RuntimeError(
                    "Could not find ANY solvable points within balance bounds.\n"
                    f"{balance.describe()}\n"
                    "This usually means the bounds are too extreme or x0 is too poor."
                )

            vals.sort(key=lambda t: t[0])
            for (x1, f1), (x2, f2) in zip(vals, vals[1:]):
                if f1 * f2 <= 0:
                    return x1, x2, f1, f2

            fs = [f for _, f in vals]
            raise RuntimeError(
                "Could not bracket target within balance bounds using solvable points.\n"
                f"{balance.describe()}\n"
                f"Solvable sample error range: min={min(fs):.6e}, max={max(fs):.6e}\n"
                "Try widening bounds or choose a different tuning knob."
            )

        expands = 0
        cur_lo, cur_hi = lo, hi
        while True:
            try:
                a, b, f_a, f_b = find_bracket(cur_lo, cur_hi)
                break
            except RuntimeError:
                expands += 1
                if expands > bracket_expand:
                    raise
                cur_lo *= 0.5
                cur_hi *= 2.0

        for _ in range(max_iter):
            mid = 0.5 * (a + b)

            ok_m, f_m, solved_m = try_err_at(mid)
            if not ok_m:
                mid = 0.5 * (mid + a)
                ok_m, f_m, solved_m = try_err_at(mid)
                if not ok_m:
                    mid = 0.5 * (mid + b)
                    ok_m, f_m, solved_m = try_err_at(mid)
                    if not ok_m:
                        raise RuntimeError(
                            "steady_state failed repeatedly near midpoint during balancing.\n"
                            f"{balance.describe()}\n"
                            f"Current bracket: [{a:.3e}, {b:.3e}]"
                        )

            if abs(f_m) < tol_y:
                return solved_m  # type: ignore

            if f_a * f_m <= 0:
                b, f_b = mid, f_m
            else:
                a, f_a = mid, f_m

        ok_m, f_m, solved_m = try_err_at(0.5 * (a + b))
        if ok_m and solved_m is not None:
            return solved_m
        raise RuntimeError(
            "Balance hit max_iter and could not produce a final solved state.\n"
            f"{balance.describe()}\n"
            f"Final bracket: [{a:.3e}, {b:.3e}]"
        )