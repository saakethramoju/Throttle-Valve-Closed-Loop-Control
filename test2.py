"""
test_teststand.py

Smoke-test for TestStand + component classes (with fixed Nozzle.__str__).

What this does:
- Instantiates a plausible set of components
- Builds a TestStand
- Prints the TestStand (your comprehensive __str__)
- Runs a tiny "fake transient" loop that ramps a valve CdA and updates mdot
  using a simple incompressible orifice relation (NOT a full network solver)

Run:
    python test_teststand.py
"""

import numpy as np

from Components import *  # assumes your classes live in Components.py


class TestStand:
    def __init__(self,
                 FuelTank: Tank, OxTank: Tank,
                 FuelRunline: Line, OxRunline: Line,
                 FuelThrottleValve: Valve, OxThrottleValve: Valve,
                 FuelInjectorManifold: InjectorManifold, OxInjectorManifold: InjectorManifold,
                 FuelInjector: Orifice, OxInjector: Orifice,
                 MainChamber: CombustionChamber, TCA: Nozzle):
        
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

    def __str__(self):
        try:
            mdot_f = self.FuelInjector.mdot
            mdot_ox = self.OxInjector.mdot
            mdot_total = mdot_f + mdot_ox
            MR = mdot_ox / mdot_f if mdot_f != 0 else float("inf")
        except Exception:
            mdot_f = mdot_ox = mdot_total = MR = float("nan")

        return (
            "\n================ TEST STAND STATE ================\n\n"

            "----- FUEL SIDE -----\n"
            f"{self.FuelTank}\n"
            f"{self.FuelRunline}\n"
            f"{self.FuelThrottleValve}\n"
            f"{self.FuelInjectorManifold}\n"
            f"{self.FuelInjector}\n\n"

            "----- OXIDIZER SIDE -----\n"
            f"{self.OxTank}\n"
            f"{self.OxRunline}\n"
            f"{self.OxThrottleValve}\n"
            f"{self.OxInjectorManifold}\n"
            f"{self.OxInjector}\n\n"

            "----- COMBUSTION -----\n"
            f"{self.MainChamber}\n"
            f"{self.TCA}\n\n"

            "----- PERFORMANCE -----\n"
            f"Fuel mdot      : {mdot_f:.3e} kg/s\n"
            f"Ox mdot        : {mdot_ox:.3e} kg/s\n"
            f"Total mdot     : {mdot_total:.3e} kg/s\n"
            f"Mixture Ratio  : {MR:.3f}\n"

            "\n===================================================\n"
        )


def mdot_incompressible(CdA: float, rho: float, p_up: float, p_down: float) -> float:
    """
    Simple incompressible flow law:
      mdot = sign(dp) * CdA * sqrt(2*rho*|dp|)
    """
    dp = p_up - p_down
    return (1.0 if dp >= 0 else -1.0) * CdA * np.sqrt(2.0 * rho * abs(dp))


def CdA_series(CdA1: float, CdA2: float) -> float:
    """
    Very rough equivalent for two CdA elements in series under
    mdot = CdA * sqrt(2*rho*dp) modeling:
        1/CdAeq^2 = 1/CdA1^2 + 1/CdA2^2
    """
    return 1.0 / np.sqrt((1.0 / CdA1**2) + (1.0 / CdA2**2))


def main():
    # -------------------------
    # Nominal constants
    # -------------------------
    PA_PER_PSI = 6894.757293168

    p_fuel_tank = 450.0 * PA_PER_PSI
    p_ox_tank   = 400.0 * PA_PER_PSI

    # arbitrary chamber pressure for smoke test
    p_chamber = 200.0 * PA_PER_PSI

    rho_fuel = 800.0    # kg/m^3
    rho_ox   = 1140.0   # kg/m^3

    # -------------------------
    # Instantiate volumes
    # -------------------------
    FuelTank = Tank("FuelTank", pressure=p_fuel_tank, volume=0.020, density=rho_fuel)
    OxTank   = Tank("OxTank",   pressure=p_ox_tank,   volume=0.020, density=rho_ox)

    FuelMan  = InjectorManifold("FuelManifold", pressure=p_fuel_tank * 0.95, volume=2.0e-4, density=rho_fuel)
    OxMan    = InjectorManifold("OxManifold",   pressure=p_ox_tank   * 0.95, volume=2.0e-4, density=rho_ox)

    MainChamber = CombustionChamber("MainChamber", pressure=p_chamber, volume=1.0e-3, mixture_ratio=2.2)

    # -------------------------
    # Instantiate branches
    # -------------------------
    FuelRunline = Line("FuelRunline", mass_flow=0.0, length=1.0, cross_sectional_area=2.0e-4, Cd=0.85)
    OxRunline   = Line("OxRunline",   mass_flow=0.0, length=1.0, cross_sectional_area=2.0e-4, Cd=0.85)

    FuelThrottleValve = Valve("FuelThrottleValve", mass_flow=0.0, CdA=0.50e-4)
    OxThrottleValve   = Valve("OxThrottleValve",   mass_flow=0.0, CdA=0.50e-4)

    FuelInjector = Orifice("FuelInjector", mass_flow=0.0, CdA=0.30e-4)
    OxInjector   = Orifice("OxInjector",   mass_flow=0.0, CdA=0.30e-4)

    # Nozzle (now that your __str__ is fixed)
    TCA = Nozzle(
        name="TCA",
        mass_flow=0.0,
        throat_area=1.0e-4,
        expansion_ratio=20.0,
        contraction_ratio=6.0,
        nfz=0
    )

    # -------------------------
    # Build TestStand
    # -------------------------
    stand = TestStand(
        FuelTank=FuelTank, OxTank=OxTank,
        FuelRunline=FuelRunline, OxRunline=OxRunline,
        FuelThrottleValve=FuelThrottleValve, OxThrottleValve=OxThrottleValve,
        FuelInjectorManifold=FuelMan, OxInjectorManifold=OxMan,
        FuelInjector=FuelInjector, OxInjector=OxInjector,
        MainChamber=MainChamber, TCA=TCA
    )

    # -------------------------
    # Initial print
    # -------------------------
    print("INITIAL STATE:")
    print(stand)

    # -------------------------
    # Tiny "fake transient" loop
    # -------------------------
    t = np.linspace(0.0, 2.0, 6)  # 0, 0.4, ..., 2.0 s
    print("RUNNING SMOKE-TEST LOOP...\n")

    for tk in t:
        # Example: ramp fuel valve opening, keep ox constant
        stand.FuelThrottleValve.CdA = 0.20e-4 + (0.80e-4) * (tk / t[-1])
        stand.OxThrottleValve.CdA   = 0.50e-4

        # Very rough series CdA for valve + injector
        CdA_f_eq  = CdA_series(stand.FuelThrottleValve.CdA, stand.FuelInjector.CdA)
        CdA_ox_eq = CdA_series(stand.OxThrottleValve.CdA,   stand.OxInjector.CdA)

        mdot_f  = mdot_incompressible(CdA_f_eq,  rho_fuel, stand.FuelTank.p, stand.MainChamber.p)
        mdot_ox = mdot_incompressible(CdA_ox_eq, rho_ox,   stand.OxTank.p,   stand.MainChamber.p)

        # Update mdot fields for display
        stand.FuelRunline.mdot = mdot_f
        stand.FuelThrottleValve.mdot = mdot_f
        stand.FuelInjector.mdot = mdot_f

        stand.OxRunline.mdot = mdot_ox
        stand.OxThrottleValve.mdot = mdot_ox
        stand.OxInjector.mdot = mdot_ox

        stand.TCA.mdot = mdot_f + mdot_ox

        MR = mdot_ox / mdot_f if mdot_f != 0 else float("inf")
        print(
            f"t={tk:4.1f} s | "
            f"Fuel CdA={stand.FuelThrottleValve.CdA:.3e} | "
            f"mdot_f={mdot_f:.3e} kg/s | "
            f"mdot_ox={mdot_ox:.3e} kg/s | MR={MR:.3f}"
        )

    print("\nFINAL STATE:")
    print(stand)


if __name__ == "__main__":
    main()