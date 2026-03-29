import numpy as np
import matplotlib.pyplot as plt
from Utilities import set_winplot_dark
from Controller import PID, step, SecondOrderActuator

set_winplot_dark()
# ============================================================
# User Inputs / Tunables
# ============================================================

# --- Simulation time grid ---
dt = 0.001                              # simulation timestep [s]
t_final = 2.0                           # total simulation time [s]

# --- Angle limits ---
angle_min = -90.0                       # actuator lower travel limit [deg]
angle_max = 90.0                        # actuator upper travel limit [deg]

# --- Actuator parameters ---
actuator_wn = 40.0                      # actuator natural frequency [rad/s]
actuator_zeta = 0.5                     # actuator damping ratio [-]
actuator_initial_angle = 0.0            # initial actuator angle [deg]
actuator_initial_velocity = 0.0         # initial actuator angular velocity [deg/s]
actuator_max_velocity = 360.0           # optional hard angular velocity limit [deg/s]

# --- Angle target schedule ---
# DEFINED BELOW

# --- PID tuning ---
Kp_angle = 100.0                          # proportional gain on angle error
Ki_angle = 200                          # integral gain on angle error
Kd_angle = 0.5                         # derivative gain on angle measurement

u_bias_angle = 0.0                      # baseline angle command sent to actuator
tau_d_angle = 0.00                      # derivative filter time constant [s]
du_dt_limit_angle = None                # optional rate limit on PID output [deg/s]


# ============================================================
# Time Grid
# ============================================================
timespan = np.arange(0.0, t_final + dt, dt)


# ============================================================
# Schedule Setup
# ============================================================
angle_target_schedule = step(
    timespan,
    initial_value=0.0,
    final_value=30.0,
    t_step=0.1,
)


# ============================================================
# Actuator Setup
# ============================================================
actuator = SecondOrderActuator(
    initial_value=actuator_initial_angle,
    min_value=angle_min,
    max_value=angle_max,
    wn=actuator_wn,
    zeta=actuator_zeta,
    initial_velocity=actuator_initial_velocity,
    max_velocity=actuator_max_velocity,
)


# ============================================================
# PID Setup
# ============================================================
pid_angle = PID(
    Kp=Kp_angle,
    Ki=Ki_angle,
    Kd=Kd_angle,
    u_min=angle_min,                    # PID output is commanded actuator angle
    u_max=angle_max,
    u_bias=u_bias_angle,
    tau_d=tau_d_angle,
    du_dt_limit=du_dt_limit_angle,
)

pid_angle.reset(
    measurement=actuator.value,
    output=actuator.value,
)


# ============================================================
# Storage
# ============================================================
angle_measured = np.zeros_like(timespan)
angle_velocity = np.zeros_like(timespan)
angle_command = np.zeros_like(timespan)

angle_error_hist = np.zeros_like(timespan)
angle_integral_hist = np.zeros_like(timespan)
angle_dmeas_hist = np.zeros_like(timespan)


# ============================================================
# Initial Storage
# ============================================================
angle_measured[0] = actuator.value
angle_velocity[0] = actuator.velocity
angle_command[0] = actuator.value

angle_error_hist[0] = angle_target_schedule[0] - actuator.value
angle_integral_hist[0] = pid_angle.integral
angle_dmeas_hist[0] = pid_angle.derivative


# ============================================================
# Closed-Loop Simulation
# ============================================================
for i, t in enumerate(timespan[:-1]):

    target_angle = angle_target_schedule[i]
    measured_angle = actuator.value

    # PID computes the commanded actuator angle
    cmd_angle, err_angle, integ_angle, dmeas_angle = pid_angle.update(
        target=target_angle,
        measurement=measured_angle,
        dt=dt,
    )

    # Actuator follows the commanded angle with second-order dynamics
    actuator.update(cmd_angle, dt)

    angle_measured[i + 1] = actuator.value
    angle_velocity[i + 1] = actuator.velocity
    angle_command[i + 1] = cmd_angle

    angle_error_hist[i + 1] = err_angle
    angle_integral_hist[i + 1] = integ_angle
    angle_dmeas_hist[i + 1] = dmeas_angle



# ============================================================
# Plotting (Cleaner Layout)
# ============================================================

tracking_colors = {
    "target": "#FFD700",
    "measured": "#00FFFF",
    "velocity": "#39FF14",
    "command": "#FF6EC7",
    "error": "#FFA500",
}

fig, axs = plt.subplots(
    4, 1,
    figsize=(10, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [2.5, 1, 1, 1]}
)

# ------------------------------------------------------------
# 1) MAIN: Angle tracking (clean, no clutter)
# ------------------------------------------------------------
axs[0].plot(
    timespan,
    angle_target_schedule,
    color=tracking_colors["target"],
    linestyle="--",
    linewidth=2,
    label="Target angle",
)

axs[0].plot(
    timespan,
    angle_measured,
    color=tracking_colors["measured"],
    linewidth=2.5,
    label="Measured angle",
)

axs[0].set_ylabel("Angle (deg)")
axs[0].set_title("Actuator Angle Tracking")
axs[0].grid(alpha=0.3)
axs[0].legend()

# ------------------------------------------------------------
# 2) Actuator velocity (shows dynamics / damping)
# ------------------------------------------------------------
axs[1].plot(
    timespan,
    angle_velocity,
    color=tracking_colors["velocity"],
    linewidth=2,
    label="Angular velocity",
)

axs[1].set_ylabel("Vel (deg/s)")
axs[1].set_title("Actuator Velocity")
axs[1].grid(alpha=0.3)
axs[1].legend()

# ------------------------------------------------------------
# 3) PID output (what you're commanding)
# ------------------------------------------------------------
axs[2].plot(
    timespan,
    angle_command,
    color=tracking_colors["command"],
    linewidth=2,
    label="Commanded angle (PID output)",
)

axs[2].set_ylabel("Cmd (deg)")
axs[2].set_title("PID Output")
axs[2].grid(alpha=0.3)
axs[2].legend()

# ------------------------------------------------------------
# 4) Error (helps tuning intuition)
# ------------------------------------------------------------
axs[3].plot(
    timespan,
    angle_error_hist,
    color=tracking_colors["error"],
    linewidth=2,
    label="Angle error",
)

axs[3].set_ylabel("Error (deg)")
axs[3].set_xlabel("Time (s)")
axs[3].set_title("Tracking Error")
axs[3].grid(alpha=0.3)
axs[3].legend()

fig.tight_layout()
plt.show()