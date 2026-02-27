import numpy as np
import matplotlib.pyplot as plt
from Constants import PA_PER_PSI, M3_PER_L, M2_PER_IN2
from FluidUtilities import incompressible_CdA_equation
from Components import *
from TestStand import TestStand


FuelTank = Tank("Fuel Tank", "RP-1", 450 * PA_PER_PSI, 70/1000, 800)
OxTank = Tank("Oxidizer Tank", "LOX", 400 *PA_PER_PSI, 70/1000, 1104)

FuelRunline = Line("Fuel Runline")
OxRunline = Line("Oxidizer Runline")

FuelThrottle = Valve("Fuel Throttle Valve", 0.5e-4)
OxThrottle = Valve("Oxidizer Throttle Valve", 1e-4)

FuelInjectorManifold = InjectorManifold("Fuel Manifold", 350 * PA_PER_PSI)
OxInjectorManifold = InjectorManifold("Oxidizer Manifold", 350 * PA_PER_PSI)

FuelInjector = Orifice("Fuel Injector", 0.5e-4)
OxInjector = Orifice("Oxidizer Injector", 1e-4)

Chamber = CombustionChamber("Main Chamber", 300 * PA_PER_PSI)
TCA = Nozzle("Nozzle", 5 * M2_PER_IN2)

Atmosphere = Drain("Ambient")

HETS = TestStand("HETS",
                 FuelTank, OxTank,
                 FuelRunline, OxRunline,
                 FuelThrottle, OxThrottle, 
                 FuelInjectorManifold, OxInjectorManifold, 
                 FuelInjector, OxInjector,
                 Chamber, TCA, Atmosphere)

print(HETS.MainChamber)

HETS = HETS.steady_state()

print(HETS)
