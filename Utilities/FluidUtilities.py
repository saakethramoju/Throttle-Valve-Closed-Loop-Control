import numpy as np
from CoolProp.CoolProp import PropsSI
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



def get_density(fluid_name: str, pressure: float, temperature: float) -> float:
    """
    Return the density of a fluid using CoolProp, after normalizing common
    alternate names to CoolProp-compatible pure fluid names.

    Parameters
    ----------
    fluid_name : str
        Fluid name or alias (e.g. 'Water', 'lox', 'RP-1', 'kerosene').
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Density in kg/m^3.
    """
    coolprop_name = normalize_fluid_name(fluid_name)
    return PropsSI("D", "P", pressure, "T", temperature, coolprop_name)

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