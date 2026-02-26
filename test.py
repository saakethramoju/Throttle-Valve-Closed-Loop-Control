import numpy as np
import matplotlib.pyplot as plt
from UnitConversions import PA_PER_PSI


# Input
P_tank_f  = 450.0 * PA_PER_PSI   # Pa
P_tank_ox = 400.0 * PA_PER_PSI   # Pa  (not used yet, kept)

rho_fuel = 800.0                 # kg/m^3
line_CdA = 0.5e-4                # m^2

t = np.linspace(0.0, 5.0, 400)   # s

# Backpressure definition in psia (for readability), then convert to Pa
P_back_psia = 200.0 + 100.0 * np.heaviside(t - 3.0, 1.0)
P_back      = P_back_psia * PA_PER_PSI   # Pa

# -----------------------------
# Utility
# -----------------------------
def mdot_from_cda(CdA, P1, P2, rho):
    """Orifice-equivalent: mdot = sign(dP) * CdA * sqrt(2*rho*|dP|)"""
    dP = P1 - P2
    return np.sign(dP) * CdA * np.sqrt(2.0 * rho * np.abs(dP))

# Mass flow (kg/s)
mdot = mdot_from_cda(line_CdA, P_tank_f, P_back, rho_fuel)

# If reverse flow is physically impossible for your case (check valve / injector),
# clamp it to 0:
mdot = np.maximum(mdot, 0.0)

# -----------------------------
# Plot (mdot + backpressure sanity)
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t, mdot, linewidth=2.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass Flow (kg/s)")
ax.grid(True)

ax2 = ax.twinx()
ax2.plot(t, P_back_psia, linestyle="--", linewidth=2.0)
ax2.set_ylabel("Back Pressure (psia)")

plt.tight_layout()
plt.show()