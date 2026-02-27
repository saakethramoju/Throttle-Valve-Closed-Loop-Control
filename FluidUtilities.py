import numpy as np

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



