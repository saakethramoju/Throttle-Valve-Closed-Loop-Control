import numpy as np
import matplotlib.pyplot as plt

from Utilities import (
    set_winplot_dark,
    get_density,
    get_pressure,
    incompressible_CdA_equation,
    create_CEA_object,
    get_chamber_pressure,
    choked_nozzle_mass_flow,
)
from Physics import PA_PER_PSI, M2_PER_IN2
from Network.Components import *

set_winplot_dark()

FuelTank = Tank("Fuel Tank", "jet-a", 550 * PA_PER_PSI, 70 / 1000, temperature=300)
OxTank = Tank("Oxidizer Tank", "LOX", 500 * PA_PER_PSI, 70 / 1000, temperature=90)

FuelInjectorManifold = InjectorManifold(
    "Fuel Manifold",
    432.54 * PA_PER_PSI,
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
    mixture_ratio=2.4173,
    cstar_efficiency=1,
    volume=6e-2,
)

Atmosphere = Drain("Ambient")

FuelThrottle = Valve("Fuel Throttle Valve", CdA=0.5e-4, mass_flow=0)
OxThrottle = Valve("Oxidizer Throttle Valve", CdA=1.0e-4, mass_flow=0)

FuelRunline = Line("Fuel Runline", length=5, cross_sectional_area=0.5e-4, mass_flow=0)
OxRunline = Line("Oxidizer Runline", length=5, cross_sectional_area=0.5e-4, mass_flow=0)

FuelInjector = Orifice("Fuel Injector", 0.5e-4, mass_flow=0)
OxInjector = Orifice("Oxidizer Injector", 1.0e-4, mass_flow=0)

TCA = Nozzle("Nozzle", 7.5313 * M2_PER_IN2, mass_flow=0)


def print_state():
    print()
    print("FUEL SIDE:")
    print(
        f"||{FuelTank.p / PA_PER_PSI:.2f} psia|| ----> "
        f"{FuelThrottle.mdot:.2f} kg/s, {FuelThrottle.CdA * 1e4:.2f} cm^2 ----> "
        f"||{FuelInjectorManifold.p / PA_PER_PSI:.2f} psia|| ----> "
        f"{FuelInjector.mdot:.2f} kg/s, {FuelInjector.CdA * 1e4:.2f} cm^2 ----> "
        f"||{Chamber.p / PA_PER_PSI:.2f} psia, {Chamber.MR:.2f}|| ----> "
        f"{TCA.mdot:.2f} kg/s ----> "
        f"||{Atmosphere.p / PA_PER_PSI:.2f} psia||"
    )
    print()
    print("OX SIDE:")
    print(
        f"||{OxTank.p / PA_PER_PSI:.2f} psia|| ----> "
        f"{OxThrottle.mdot:.2f} kg/s, {OxThrottle.CdA * 1e4:.2f} cm^2 ----> "
        f"||{OxInjectorManifold.p / PA_PER_PSI:.2f} psia|| ----> "
        f"{OxInjector.mdot:.2f} kg/s, {OxInjector.CdA * 1e4:.2f} cm^2 ----> "
        f"||{Chamber.p / PA_PER_PSI:.2f} psia, {Chamber.MR:.2f}|| ----> "
        f"{TCA.mdot:.2f} kg/s ----> "
        f"||{Atmosphere.p / PA_PER_PSI:.2f} psia||"
    )
    print()


print_state()

dt = 0.001
timespan = np.arange(0, 10 + dt, dt)

# =========================
# Fuel throttle schedule
# =========================
fuel_CdA_initial = FuelThrottle.CdA
fuel_CdA_final = 2.0 * fuel_CdA_initial

fuel_t1 = 1.5
fuel_t2 = 3.0

fuel_throttle_CdA = np.zeros_like(timespan)

for i, t in enumerate(timespan):
    if t < fuel_t1:
        fuel_throttle_CdA[i] = fuel_CdA_initial
    elif t < fuel_t2:
        frac = (t - fuel_t1) / (fuel_t2 - fuel_t1)
        fuel_throttle_CdA[i] = fuel_CdA_initial + frac * (fuel_CdA_final - fuel_CdA_initial)
    else:
        fuel_throttle_CdA[i] = fuel_CdA_final

# =========================
# Ox throttle schedule
# =========================
ox_CdA_initial = 0.95 * OxThrottle.CdA
ox_CdA_final = 1.15 * ox_CdA_initial

ox_t1 = 1.5
ox_t2 = 3.0

ox_throttle_CdA = np.zeros_like(timespan)

for i, t in enumerate(timespan):
    if t < ox_t1:
        ox_throttle_CdA[i] = ox_CdA_initial
    elif t < ox_t2:
        frac = (t - ox_t1) / (ox_t2 - ox_t1)
        ox_throttle_CdA[i] = ox_CdA_initial + frac * (ox_CdA_final - ox_CdA_initial)
    else:
        ox_throttle_CdA[i] = ox_CdA_final



initial_overall_mdot = choked_nozzle_mass_flow(
    Chamber.p,
    Chamber.MR,
    TCA.At,
    Chamber.eta_cstar,
)
initial_fuel_mdot = initial_overall_mdot / (1 + Chamber.MR)
initial_ox_mdot = initial_overall_mdot * Chamber.MR / (1 + Chamber.MR)

cea_obj = create_CEA_object()

fuel_sys_L = FuelRunline.L / FuelRunline.A
ox_sys_L = OxRunline.L / OxRunline.A

FuelThrottle.mdot = initial_fuel_mdot
FuelRunline.mdot = initial_fuel_mdot
FuelInjector.mdot = initial_fuel_mdot

OxThrottle.mdot = initial_ox_mdot
OxRunline.mdot = initial_ox_mdot
OxInjector.mdot = initial_ox_mdot

TCA.mdot = initial_overall_mdot

fuel_sys_mdot = np.zeros(timespan.size)
ox_sys_mdot = np.zeros(timespan.size)

fuel_inj_pressure = np.zeros(timespan.size)
ox_inj_pressure = np.zeros(timespan.size)

fuel_inj_mdot = np.zeros(timespan.size)
ox_inj_mdot = np.zeros(timespan.size)

chamber_pressure = np.zeros(timespan.size)
mixture_ratio = np.zeros(timespan.size)
tca_mdot = np.zeros(timespan.size)

fuel_sys_mdot[0] = FuelRunline.mdot
ox_sys_mdot[0] = OxRunline.mdot

fuel_inj_pressure[0] = FuelInjectorManifold.p
ox_inj_pressure[0] = OxInjectorManifold.p

fuel_inj_mdot[0] = FuelInjector.mdot
ox_inj_mdot[0] = OxInjector.mdot

chamber_pressure[0] = Chamber.p
mixture_ratio[0] = Chamber.MR
tca_mdot[0] = TCA.mdot

for i, _ in enumerate(timespan[:-1]):
    # Apply commanded throttles at current time step
    FuelThrottle.CdA = fuel_throttle_CdA[i]
    OxThrottle.CdA = ox_throttle_CdA[i]

    # Current resistances
    fuel_sys_R = 1 / (2 * FuelThrottle.CdA**2)
    ox_sys_R = 1 / (2 * OxThrottle.CdA**2)

    # Current densities at step i
    fuel_tank_rho = get_density(FuelTank.propellant, FuelTank.p, FuelTank.T)
    ox_tank_rho = get_density(OxTank.propellant, OxTank.p, OxTank.T)

    fuel_inj_rho = get_density(
        FuelTank.propellant,
        fuel_inj_pressure[i],
        FuelInjectorManifold.T,
    )
    ox_inj_rho = get_density(
        OxTank.propellant,
        ox_inj_pressure[i],
        OxInjectorManifold.T,
    )

    # Derivatives evaluated strictly at step i
    fuel_sys_dm_dt = (
        (FuelTank.p - fuel_inj_pressure[i])
        - (fuel_sys_R / fuel_tank_rho) * fuel_sys_mdot[i] ** 2
    ) / fuel_sys_L

    ox_sys_dm_dt = (
        (OxTank.p - ox_inj_pressure[i])
        - (ox_sys_R / ox_tank_rho) * ox_sys_mdot[i] ** 2
    ) / ox_sys_L

    fuel_inj_mdot_i = incompressible_CdA_equation(
        fuel_inj_pressure[i],
        chamber_pressure[i],
        fuel_inj_rho,
        FuelInjector.CdA,
    )
    ox_inj_mdot_i = incompressible_CdA_equation(
        ox_inj_pressure[i],
        chamber_pressure[i],
        ox_inj_rho,
        OxInjector.CdA,
    )

    mr_i = ox_inj_mdot_i / fuel_inj_mdot_i

    tca_mdot_i = choked_nozzle_mass_flow(
        chamber_pressure[i],
        mr_i,
        TCA.At,
        Chamber.eta_cstar,
    )

    fuel_inj_drho_dt = (fuel_sys_mdot[i] - fuel_inj_mdot_i) / FuelInjectorManifold.V
    ox_inj_drho_dt = (ox_sys_mdot[i] - ox_inj_mdot_i) / OxInjectorManifold.V

    chamber_rho_i = cea_obj.get_Chamber_Density(chamber_pressure[i], mixture_ratio[i])
    chamber_drho_dt = (fuel_inj_mdot_i + ox_inj_mdot_i - tca_mdot_i) / Chamber.V

    # Forward Euler updates
    fuel_sys_mdot[i + 1] = fuel_sys_mdot[i] + fuel_sys_dm_dt * dt
    ox_sys_mdot[i + 1] = ox_sys_mdot[i] + ox_sys_dm_dt * dt

    fuel_inj_mdot[i + 1] = fuel_inj_mdot_i
    ox_inj_mdot[i + 1] = ox_inj_mdot_i
    mixture_ratio[i + 1] = mr_i
    tca_mdot[i + 1] = tca_mdot_i

    new_fuel_inj_rho = fuel_inj_rho + fuel_inj_drho_dt * dt
    new_ox_inj_rho = ox_inj_rho + ox_inj_drho_dt * dt
    new_chamber_rho = chamber_rho_i + chamber_drho_dt * dt

    fuel_inj_pressure[i + 1] = get_pressure(
        FuelTank.propellant,
        new_fuel_inj_rho,
        FuelInjectorManifold.T,
    )
    ox_inj_pressure[i + 1] = get_pressure(
        OxTank.propellant,
        new_ox_inj_rho,
        OxInjectorManifold.T,
    )
    chamber_pressure[i + 1] = get_chamber_pressure(new_chamber_rho, mr_i)

    # Push updated state into component objects
    FuelInjectorManifold.p = fuel_inj_pressure[i + 1]
    OxInjectorManifold.p = ox_inj_pressure[i + 1]

    Chamber.p = chamber_pressure[i + 1]
    Chamber.MR = mixture_ratio[i + 1]

    FuelThrottle.mdot = fuel_sys_mdot[i + 1]
    OxThrottle.mdot = ox_sys_mdot[i + 1]

    FuelRunline.mdot = fuel_sys_mdot[i + 1]
    OxRunline.mdot = ox_sys_mdot[i + 1]

    FuelInjector.mdot = fuel_inj_mdot[i + 1]
    OxInjector.mdot = ox_inj_mdot[i + 1]

    TCA.mdot = tca_mdot[i + 1]

print_state()

# =========================
# Plot results
# =========================

fuel_colors = {
    "sys_mdot": "#00FFFF",
    "inj_pressure": "#FF00FF",
    "inj_mdot": "#39FF14",
}

ox_colors = {
    "sys_mdot": "#FF3131",
    "inj_pressure": "#00BFFF",
    "inj_mdot": "#FF6EC7",
}

chamber_colors = {
    "pressure": "#FFD700",
    "mr": "#39FF14",
    "tca_mdot": "#00FFFF",
}

fuel_sched_color = "#FFFFFF"
ox_sched_color = "#FFA500"

fuel_throttle_cm2 = fuel_throttle_CdA * 1e4
ox_throttle_cm2 = ox_throttle_CdA * 1e4

fuel_inj_pressure_psia = fuel_inj_pressure / PA_PER_PSI
ox_inj_pressure_psia = ox_inj_pressure / PA_PER_PSI
chamber_pressure_psia = chamber_pressure / PA_PER_PSI


def add_throttle_overlay(ax):
    ax2 = ax.twinx()
    ax2.plot(timespan, fuel_throttle_cm2, color=fuel_sched_color, linestyle="--", linewidth=1.5)
    ax2.plot(timespan, ox_throttle_cm2, color=ox_sched_color, linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Throttle CdA (cm²)")
    return ax2


# -------------------------
# Figure 1: Fuel side
# -------------------------
fig_fuel, axs_fuel = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs_fuel = axs_fuel.flatten()

axs_fuel[0].plot(timespan, fuel_sys_mdot, color=fuel_colors["sys_mdot"], linewidth=2)
axs_fuel[0].set_ylabel("Fuel System mdot (kg/s)")
axs_fuel[0].set_title("Fuel System Mass Flow")
axs_fuel[0].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[0])

axs_fuel[1].plot(timespan, fuel_inj_pressure_psia, color=fuel_colors["inj_pressure"], linewidth=2)
axs_fuel[1].set_ylabel("Fuel Injector P (psia)")
axs_fuel[1].set_title("Fuel Injector Pressure")
axs_fuel[1].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[1])

axs_fuel[2].plot(timespan, fuel_inj_mdot, color=fuel_colors["inj_mdot"], linewidth=2)
axs_fuel[2].set_ylabel("Fuel Injector mdot (kg/s)")
axs_fuel[2].set_xlabel("Time (s)")
axs_fuel[2].set_title("Fuel Injector Mass Flow")
axs_fuel[2].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[2])

axs_fuel[3].plot(timespan, fuel_throttle_cm2, color=fuel_sched_color, linewidth=2, label="Fuel")
axs_fuel[3].plot(timespan, ox_throttle_cm2, color=ox_sched_color, linewidth=2, label="Ox")
axs_fuel[3].set_ylabel("Throttle CdA (cm²)")
axs_fuel[3].set_xlabel("Time (s)")
axs_fuel[3].set_title("Throttle Schedules")
axs_fuel[3].grid(alpha=0.3)
axs_fuel[3].legend()

fig_fuel.tight_layout()

# -------------------------
# Figure 2: Ox side
# -------------------------
fig_ox, axs_ox = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_ox[0].plot(timespan, ox_sys_mdot, color=ox_colors["sys_mdot"], linewidth=2)
axs_ox[0].set_ylabel("Ox System mdot (kg/s)")
axs_ox[0].set_title("Oxidizer System Mass Flow")
axs_ox[0].grid(alpha=0.3)
add_throttle_overlay(axs_ox[0])

axs_ox[1].plot(timespan, ox_inj_pressure_psia, color=ox_colors["inj_pressure"], linewidth=2)
axs_ox[1].set_ylabel("Ox Injector P (psia)")
axs_ox[1].set_title("Oxidizer Injector Pressure")
axs_ox[1].grid(alpha=0.3)
add_throttle_overlay(axs_ox[1])

axs_ox[2].plot(timespan, ox_inj_mdot, color=ox_colors["inj_mdot"], linewidth=2)
axs_ox[2].set_ylabel("Ox Injector mdot (kg/s)")
axs_ox[2].set_xlabel("Time (s)")
axs_ox[2].set_title("Oxidizer Injector Mass Flow")
axs_ox[2].grid(alpha=0.3)
add_throttle_overlay(axs_ox[2])

fig_ox.tight_layout()

# -------------------------
# Figure 3: Chamber / nozzle
# -------------------------
fig_chamber, axs_chamber = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_chamber[0].plot(timespan, chamber_pressure_psia, color=chamber_colors["pressure"], linewidth=2)
axs_chamber[0].set_ylabel("Chamber P (psia)")
axs_chamber[0].set_title("Chamber Pressure")
axs_chamber[0].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[0])

axs_chamber[1].plot(timespan, mixture_ratio, color=chamber_colors["mr"], linewidth=2)
axs_chamber[1].set_ylabel("Mixture Ratio")
axs_chamber[1].set_title("Mixture Ratio")
axs_chamber[1].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[1])

axs_chamber[2].plot(timespan, tca_mdot, color=chamber_colors["tca_mdot"], linewidth=2)
axs_chamber[2].set_ylabel("TCA mdot (kg/s)")
axs_chamber[2].set_xlabel("Time (s)")
axs_chamber[2].set_title("Nozzle Mass Flow")
axs_chamber[2].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[2])

fig_chamber.tight_layout()

# Show all figures together
plt.show()