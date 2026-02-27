import numpy as np
import matplotlib.pyplot as plt
from Constants import PA_PER_PSI, M3_PER_L, M2_PER_IN2
from FluidUtilities import incompressible_CdA_equation
from Components import Tank, Line, Drain
from TestStand import TestStand


FuelTank = Tank('F-Tank', 450 * PA_PER_PSI, 70 * M3_PER_L, 800)

FuelRunline = Line('Fuel Runline', 0, 3, 0.5 * M2_PER_IN2, 0.2)

Ambient = Drain("Ambient", 14.67 * PA_PER_PSI)



mdot = incompressible_CdA_equation(FuelTank.p, Ambient.p, FuelTank.rho, FuelRunline.CdA)
FuelRunline.mdot = mdot

print(FuelRunline.mdot)


