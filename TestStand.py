from Components import *

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
        except:
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