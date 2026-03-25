'''
        Pc_target, MR_target
                 │
                 ▼
            [ Error ]
                 │
        e_Pc            e_MR
         │               │
         ▼               ▼
     [ PID_Pc ]     [ PID_MR ]
         │               │
         └───────┬───────┘
                 ▼
          v = [v_Pc, v_MR]
                 │
                 ▼
          [ Mixing Matrix M ]
                 │
     u_cmd = u0 + M v
     │                     │
     ▼                     ▼
Fuel Actuator        Ox Actuator
     │                     │
     ▼                     ▼
CdA_f (actual)      CdA_ox (actual)
         └───────┬────────┘
                 ▼
               [ Plant ]
                 │
         Pc, MR (measured)
                 │
                 └─────────── feedback ───────────┘
'''

import copy
import numpy as np
import matplotlib.pyplot as plt

from Utilities import set_winplot_dark
from Physics import PA_PER_PSI, M2_PER_IN2, LBF_PER_N
from Network.Components import *
from Network import TestStand, Balance
from Controller import TestActuator, PID

set_winplot_dark()


# ============================================================
# Time Grid
# ============================================================
dt = 0.001
timespan = np.arange(0.0, 10.0 + dt, dt)


# ============================================================
# Helper Functions
# ============================================================
def ramp_hold_schedule(timespan, initial_value, final_value, t1, t2):
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1.")

    values = np.zeros_like(timespan, dtype=float)

    for i, t in enumerate(timespan):
        if t < t1:
            values[i] = initial_value
        elif t < t2:
            frac = (t - t1) / (t2 - t1)
            values[i] = initial_value + frac * (final_value - initial_value)
        else:
            values[i] = final_value

    return values


# ============================================================
# Plant Setup
# ============================================================
FuelTank = Tank("Fuel Tank", "jet-a", 450 * PA_PER_PSI, 70 / 1000, temperature=300)
OxTank = Tank("Oxidizer Tank", "LOX", 400 * PA_PER_PSI, 70 / 1000, temperature=90)

FuelInjectorManifold = InjectorManifold(
    "Fuel Manifold",
    350 * PA_PER_PSI,
    temperature=500,
    volume=0.1287,
)
OxInjectorManifold = InjectorManifold(
    "Oxidizer Manifold",
    350 * PA_PER_PSI,
    temperature=90,
    volume=0.1287,
)

Chamber = CombustionChamber(
    "Main Chamber",
    300 * PA_PER_PSI,
    mixture_ratio=2.0,
    cstar_efficiency=1,
    volume=6e-2,
)

Atmosphere = Drain("Ambient")

FuelThrottle = Valve("Fuel Throttle Valve", CdA=0.5e-4)
OxThrottle = Valve("Oxidizer Throttle Valve", CdA=1.0e-4)

FuelRunline = Line("Fuel Runline", length=5, cross_sectional_area=0.5e-4)
OxRunline = Line("Oxidizer Runline", length=5, cross_sectional_area=0.5e-4)

FuelInjector = Orifice("Fuel Injector", 0.5e-4)
OxInjector = Orifice("OxInjector", 1.0e-4)

TCA = Nozzle(
    "Nozzle",
    throat_area=6.05 * M2_PER_IN2,
    contraction_ratio=2,
    expansion_ratio=4.7,
    eta_cf=0.95,
    nfz=2,
)

HETS = TestStand(
    "HETS",
    FuelTank, OxTank,
    FuelRunline, OxRunline,
    FuelThrottle, OxThrottle,
    FuelInjectorManifold, OxInjectorManifold,
    FuelInjector, OxInjector,
    Chamber, TCA, Atmosphere
)


# --- Build Steady-State Initial Condition ---
ts = copy.deepcopy(HETS)

#ss_result = ts.steady_state()
MR_balance = Balance(
    tune="OxThrottleValve.CdA",
    measure="MainChamber.MR",
    target=2.3,
    bounds=(1e-6, 1e-4),
    tol=1e-5,   #
)
ss_result = ts.steady_state_with_balance(MR_balance)
if ss_result is not None:
    ts = ss_result

print(ts)