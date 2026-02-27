class Fluid:
    """
    Represents a single-phase fluid with uniform thermodynamic properties.

    This class is intended for use in quasi-steady propulsion and fluid
    system simulations (e.g., tanks, feed lines, injectors, nozzles).
    It assumes spatially uniform properties (lumped-parameter model).

    Parameters
    ----------
    name : str
        Identifier for the fluid (e.g., 'LOX', 'RP-1', 'Water').
    pressure : float
        Static pressure of the fluid [Pa]. Must be > 0.
    density : float
        Fluid density [kg/m^3]. Must be > 0.

    Attributes
    ----------
    name : str
        Fluid identifier.
    p : float
        Static pressure [Pa].
    rho : float
        Density [kg/m^3].

    Notes
    -----
    - This model does not account for temperature, compressibility,
      or phase change.
    - For cryogenic or high-speed flow modeling, additional
      thermodynamic properties may be required.
    - Units are SI unless otherwise specified.

    Examples
    --------
    >>> fuel = Fluid('RP-1', pressure=5e6, density=810)
    >>> oxidizer = Fluid('LOX', pressure=6e6, density=1140)
    """

    def __init__(self,
                 name: str,
                 pressure: float,
                 density: float):

        if pressure <= 0:
            raise ValueError("pressure must be > 0 Pa")
        if density <= 0:
            raise ValueError("density must be > 0 kg/m^3")

        self.name = name
        self.p = float(pressure)      # Pa
        self.rho = float(density)     # kg/m^3

    def __str__(self) -> str:
        return (
            f"Fluid(name='{self.name}', "
            f"p={self.p:.3e} Pa, "
            f"rho={self.rho:.3f} kg/m^3)"
        )
    
water = Fluid('Water', 101325, 999.84283)