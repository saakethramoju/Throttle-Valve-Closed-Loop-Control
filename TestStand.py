from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import root

from Components import *
from CombustionUtilities import choked_nozzle_thrust, choked_nozzle_mass_flow, create_CEA_object
from FluidUtilities import incompressible_CdA_equation


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
        Feed lines. 
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
    """

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

    def __repr__(self) -> str:
        return f"TestStand (name={self.name}, FuelTank={self.FuelTank!r}, OxTank={self.OxTank!r}, MainChamber={self.MainChamber!r}, TCA={self.TCA!r})"


    def __str__(self) -> str:
        # Local helpers
        def _fmt_pa(p):
            return f"{p:,.0f} Pa" if p is not None else "—"

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

        # Build tables
        state_rows = [
            ("Fuel Tank",        _fmt_pa(P_tank_f),   f"{rho_f:.1f} kg/m^3" if rho_f is not None else "—"),
            ("Fuel Manifold",    _fmt_pa(P_inj_f),    f"{rho_f_man:.1f} kg/m^3" if rho_f_man is not None else "—"),
            ("Ox Tank",          _fmt_pa(P_tank_ox),  f"{rho_ox:.1f} kg/m^3" if rho_ox is not None else "—"),
            ("Ox Manifold",      _fmt_pa(P_inj_ox),   f"{rho_ox_man:.1f} kg/m^3" if rho_ox_man is not None else "—"),
            ("Chamber",          _fmt_pa(Pc),         "—"),
            ("Ambient",          _fmt_pa(Pamb),       "—"),
        ]

        flow_rows = [
            ("Fuel mdot",  _fmt_kgps(mdot_f)),
            ("Ox mdot",    _fmt_kgps(mdot_ox)),
            ("Total mdot", _fmt_kgps(mdot_total)),
            ("MR",   f"{MR:.4f}" if MR is not None else "—"),
        ]

        geom_rows = [
            ("At",  f"{At:.3e} m^2" if At is not None else "—"),
            ("eps", f"{eps:.3f}" if eps is not None else "—"),
            ("F",   f"{F:,.2f} N" if F is not None else "—"),
        ]

        cda_rows = [
            ("Fuel throttle CdA", f"{CdA_f_sys:.3e} m^2" if CdA_f_sys is not None else "—"),
            ("Ox throttle CdA",   f"{CdA_ox_sys:.3e} m^2" if CdA_ox_sys is not None else "—"),
            ("Fuel injector CdA", f"{CdA_f_inj:.3e} m^2" if CdA_f_inj is not None else "—"),
            ("Ox injector CdA",   f"{CdA_ox_inj:.3e} m^2" if CdA_ox_inj is not None else "—"),
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




    def steady_state(
        self,
        x0: list[float] | None = None,
        tol: float = 1e-9,
        maxfev: int | None = None,
        Pamb: float | None = None,
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
        If ambient pressure is available (via `Pamb` argument or `self.Ambient.p`), it also
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
        Pamb : float or None, optional
            Ambient/back pressure [Pa] used only for thrust computation. If None, this
            method will try to use `self.Ambient.p` if it exists; otherwise thrust is
            not computed.

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
        cea_obj = create_CEA_object(fuel_name, ox_name)

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
            if inj_fuel_mdot <= 0:
                MR = 1e12
            else:
                MR = inj_ox_mdot / inj_fuel_mdot

            # Use the correct function name in your codebase:
            # If your helper is named choked_mass_flow, use that.
            tca_mdot = choked_nozzle_mass_flow(Pc, MR, At, eta_cstar, cea_obj)

            return np.array([
                sys_fuel_mdot - inj_fuel_mdot,
                sys_ox_mdot - inj_ox_mdot,
                (inj_fuel_mdot + inj_ox_mdot) - tca_mdot
            ], dtype=float)

        if x0 is None:
            x0 = [
                float(self.FuelInjectorManifold.p),
                float(self.OxInjectorManifold.p),
                float(self.MainChamber.p)
            ]

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

        # Nozzle state
        solved.TCA.mdot = solved_fuel_mdot + solved_ox_mdot

        # Thrust (optional)
        if Pamb is None:
            if hasattr(self, "Ambient") and hasattr(self.Ambient, "p"):
                Pamb = float(self.Ambient.p)

        if Pamb is not None:
            solved.TCA.F = choked_nozzle_thrust(
                Pc_sol, MR_sol, solved.TCA.At, float(Pamb),
                solved.TCA.eps, solved.TCA.eta_cf, solved.TCA.nfz, cea_obj
            )

        return solved