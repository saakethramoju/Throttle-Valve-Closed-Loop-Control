import numpy as np
from functools import lru_cache
from CoolProp.CoolProp import PropsSI
from scipy.optimize import toms748
import warnings
from .FLUID_NAMES import FLUID_NAME_BANK

def normalize_fluid_name(fluid_name: str) -> str:
    """
    Convert a user-provided fluid name or alias into a CoolProp-compatible
    pure fluid name.

    The input is lowercased and stripped before lookup.

    Parameters
    ----------
    fluid_name : str
        User-provided fluid name.

    Returns
    -------
    str
        CoolProp-compatible fluid name.

    Raises
    ------
    TypeError
        If fluid_name is not a string.
    ValueError
        If fluid_name is empty.
    KeyError
        If no matching CoolProp fluid name or alias is found.
    """
    if not isinstance(fluid_name, str):
        raise TypeError("fluid_name must be a string.")

    key = fluid_name.strip().lower()

    if not key:
        raise ValueError("fluid_name cannot be empty.")

    # 🔑 Lookup in alias bank
    if key in FLUID_NAME_BANK:
        return FLUID_NAME_BANK[key]

    # 🔁 Fallback: try direct CoolProp name
    try:
        PropsSI("D", "P", 101325.0, "T", 300.0, fluid_name)
        return fluid_name
    except Exception:
        pass

    raise KeyError(
        f"Unknown fluid '{fluid_name}'. Add it to fluid_name_bank.py if needed."
    )


from functools import lru_cache
from CoolProp.CoolProp import PropsSI
from scipy.optimize import toms748
import warnings


@lru_cache(maxsize=None)
def _normalized_fluid_name(fluid_name: str) -> str:
    """
    Cached normalized fluid name.
    """
    return normalize_fluid_name(fluid_name)


@lru_cache(maxsize=None)
def _critical_properties(coolprop_name: str) -> tuple[float, float]:
    """
    Cached critical temperature and pressure.

    Returns
    -------
    tuple[float, float]
        (T_crit [K], P_crit [Pa])
    """
    return (
        PropsSI("Tcrit", coolprop_name),
        PropsSI("Pcrit", coolprop_name),
    )


@lru_cache(maxsize=None)
def _saturation_properties(coolprop_name: str, temperature: float) -> tuple[float, float, float]:
    """
    Cached saturation properties at a given temperature.

    Returns
    -------
    tuple[float, float, float]
        (P_sat [Pa], rho_sat_liq [kg/m^3], rho_sat_vap [kg/m^3])
    """
    return (
        PropsSI("P", "T", temperature, "Q", 0, coolprop_name),
        PropsSI("D", "T", temperature, "Q", 0, coolprop_name),
        PropsSI("D", "T", temperature, "Q", 1, coolprop_name),
    )


def get_density(
    fluid_name: str,
    pressure: float,
    temperature: float,
    quality: float | None = 0.0,
    sat_rtol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Return the density of a fluid using CoolProp, after normalizing common
    alternate names to CoolProp-compatible pure fluid names.

    For pure fluids, density is not uniquely determined by (P, T) exactly on the
    saturation line. In that case, this function uses the provided quality factor.

    Parameters
    ----------
    fluid_name : str
        Fluid name or alias (e.g. 'Water', 'lox', 'RP-1', 'kerosene').
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    quality : float | None, optional
        Vapor quality used when (P, T) lies on the saturation line.

        - 0.0 -> saturated liquid density
        - 1.0 -> saturated vapor density
        - 0 < quality < 1 -> two-phase mixture density
        - None -> do not resolve saturation ambiguity; call PropsSI directly

        Default is 0.0.
    sat_rtol : float, optional
        Relative tolerance used to determine whether the given pressure is at
        saturation for the specified temperature.
    verbose : bool, optional
        If True, emit warnings/messages for saturation handling.

    Returns
    -------
    float
        Density in kg/m^3.

    Raises
    ------
    ValueError
        If inputs are invalid or the quality is outside [0, 1].
    """
    if pressure <= 0.0:
        raise ValueError("pressure must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if quality is not None and not (0.0 <= quality <= 1.0):
        raise ValueError("quality must be between 0 and 1, or None.")

    coolprop_name = _normalized_fluid_name(fluid_name)
    T_crit, _ = _critical_properties(coolprop_name)

    if temperature < T_crit and quality is not None:
        try:
            P_sat, _, _ = _saturation_properties(coolprop_name, temperature)
            scale = max(abs(P_sat), abs(pressure), 1.0)

            if abs(pressure - P_sat) <= sat_rtol * scale:
                if verbose:
                    warnings.warn(
                        "State lies on the saturation line; using the provided quality.",
                        RuntimeWarning,
                    )
                return PropsSI("D", "P", pressure, "Q", quality, coolprop_name)
        except Exception:
            pass

    return PropsSI("D", "P", pressure, "T", temperature, coolprop_name)


def get_pressure(
    fluid_name: str,
    density: float,
    temperature: float,
    tol: float = 1e-6,
    density_rtol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Return pressure [Pa] from density [kg/m^3] and temperature [K] for a pure fluid
    using CoolProp, with explicit handling of saturation and critical behavior.

    This function is intended for pure-fluid states where density and temperature
    are known and pressure must be recovered numerically. For subcritical
    temperatures, it explicitly checks the saturation state at the specified
    temperature and handles the following cases:

    - density equal to saturated liquid density -> return saturation pressure
    - density equal to saturated vapor density -> return saturation pressure
    - density between saturated vapor and saturated liquid density -> interpret
      the state as two-phase and return saturation pressure
    - density below saturated vapor density -> solve on the vapor branch
    - density above saturated liquid density -> solve on the compressed-liquid branch

    For critical and supercritical temperatures, the function solves directly on
    a single branch using a bracketed root finder.

    A small pressure offset is applied when solving just above or below the
    saturation line so that CoolProp is not queried exactly at saturation,
    which can otherwise raise errors for (P, T) inputs that are numerically too
    close to the coexistence boundary.

    Parameters
    ----------
    fluid_name : str
        Fluid name or alias.
    density : float
        Density in kg/m^3.
    temperature : float
        Temperature in K.
    tol : float, optional
        Absolute pressure tolerance passed to the root solver.
    density_rtol : float, optional
        Relative tolerance used for saturation-density comparisons.
    verbose : bool, optional
        If True, emit warnings/messages for two-phase handling.

    Returns
    -------
    float
        Pressure in Pa.

    Raises
    ------
    ValueError
        If inputs are invalid or no physically valid pressure can be found.
    """
    if density <= 0.0:
        raise ValueError("density must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    coolprop_name = _normalized_fluid_name(fluid_name)
    T_crit, P_crit = _critical_properties(coolprop_name)

    # Margin used to move off the saturation line before querying CoolProp
    # on the vapor or compressed-liquid branches.
    sat_pressure_margin = 1e-5

    def rho_from_PT(P: float) -> float:
        return PropsSI("D", "P", P, "T", temperature, coolprop_name)

    def residual(P: float) -> float:
        return rho_from_PT(P) - density

    def is_close(a: float, b: float, rtol: float = density_rtol) -> bool:
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= rtol * scale

    # ---------------------------------------------------------
    # 1) Subcritical temperature: explicitly handle saturation
    # ---------------------------------------------------------
    if temperature < T_crit:
        try:
            P_sat, rho_sat_liq, rho_sat_vap = _saturation_properties(
                coolprop_name,
                temperature,
            )

            if is_close(density, rho_sat_liq):
                return P_sat

            if is_close(density, rho_sat_vap):
                return P_sat

            if rho_sat_vap < density < rho_sat_liq:
                if verbose:
                    warnings.warn(
                        "Requested (rho, T) lies in the two-phase region for a pure fluid. "
                        "Returning saturation pressure. Density alone does not determine quality.",
                        RuntimeWarning,
                    )
                return P_sat

            # Vapor branch
            if density < rho_sat_vap:
                P_low = 1.0
                P_high = P_sat * (1.0 - sat_pressure_margin)

                f_low = residual(P_low)
                f_high = residual(P_high)

                if f_low * f_high > 0:
                    P_low = 1e-6
                    f_low = residual(P_low)

                if f_low * f_high > 0:
                    raise ValueError(
                        f"Could not bracket vapor-phase pressure root for "
                        f"density={density} kg/m^3 at T={temperature} K."
                    )

                return toms748(residual, P_low, P_high, xtol=tol)

            # Compressed-liquid branch
            P_low = P_sat * (1.0 + sat_pressure_margin)
            P_high = max(10.0 * P_crit, 10.0 * P_sat)

            f_low = residual(P_low)
            f_high = residual(P_high)

            n_expand = 0
            while f_low * f_high > 0 and n_expand < 20:
                P_high *= 2.0
                f_high = residual(P_high)
                n_expand += 1

            if f_low * f_high > 0:
                raise ValueError(
                    f"Could not bracket liquid-phase pressure root for "
                    f"density={density} kg/m^3 at T={temperature} K."
                )

            return toms748(residual, P_low, P_high, xtol=tol)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed while handling subcritical saturation behavior for "
                f"{fluid_name} at T={temperature} K: {e}"
            )

    # ---------------------------------------------------------
    # 2) Critical or supercritical temperature
    # ---------------------------------------------------------
    P_low = 1.0
    P_high = max(10.0 * P_crit, 1e7)

    try:
        f_low = residual(P_low)
        f_high = residual(P_high)
    except Exception as e:
        raise ValueError(
            f"Failed evaluating EOS for {fluid_name} at T={temperature} K: {e}"
        )

    n_expand = 0
    while f_low * f_high > 0 and n_expand < 25:
        P_high *= 2.0
        f_high = residual(P_high)
        n_expand += 1

    if f_low * f_high > 0:
        raise ValueError(
            f"Could not bracket pressure root for density={density} kg/m^3 "
            f"at T={temperature} K (critical/supercritical region)."
        )

    return toms748(residual, P_low, P_high, xtol=tol)



def incompressible_CdA_equation(P1: float, P2: float,   # Pa
                                rho: float,             # kg/m^3
                                CdA: float,             # m^2
                                ) -> float:
    """
    Calculates mass flow for incompressible flow using the CdA equation.

    Parameters
    ----------
    P1 : float
        Upstream pressure [Pa].
    P2 : float
        Downstream pressure [Pa].
    rho : float
        Density [kg/m^3].
    CdA : float
        Effective flow area [m^2].

    Returns
    -------
    float
        Mass flow rate [kg/s].
    """
    return np.sign(P1 - P2) * CdA * np.sqrt(2 * rho * np.abs(P1 - P2))