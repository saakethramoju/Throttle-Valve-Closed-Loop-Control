import numpy as np

# ----- Base Classes -----
class Source:
    """
    Source class represents a fixed pressure source
    
    Parameters
    ----------
    name : str
        Identifier for the source.
    pressure : float
        Static pressure within the source [Pa].
    """
    def __init__(self, 
                 name: str, 
                 pressure: float
                 ):
        
        self.name = name
        self.p = pressure

    def __str__(self):
        return f"Source (name={self.name}, p={self.p:.3e} Pa)"


class Drain:
    """
    Drain class represents a fixed pressure sink
    
    Parameters
    ----------
    name : str
        Identifier for the drain.
    pressure : float
        Static pressure within the drain [Pa].
    """
    def __init__(self, 
                 name: str, 
                 pressure: float
                 ):
        
        self.name = name
        self.p = pressure

    def __str__(self):
        return f"Drain (name={self.name}, p={self.p:.3e} Pa)"


class Volume:
    """
    Base class representing a lumped control volume in a fluid network.

    A Volume stores mass and energy and is characterized by a pressure and
    a geometric volume.
    
    Parameters
    ----------
    name : str
        Identifier for the volume.
    pressure : float
        Static pressure within the volume [Pa].
    volume : float
        Geometric volume of the control volume [m^3].
    """
    def __init__(self,
                 name: str,
                 pressure: float,   # Pa
                 volume: float      # m^3
                 ):
        
        self.name = name
        self.p = pressure
        self.V = volume

    def __str__(self):
        return f"Volume (name={self.name}, p={self.p:.3e} Pa, V={self.V:.3e} m^3)"




class Branch:
    """
    Base class representing a lumped flow element connecting two volumes.

    A Branch models a one-dimensional flow path that carries mass flow
    between volumes.
    
    Parameters
    ----------
    name : str
        Identifier for the branch.
    mass_flow : float
        Mass flow rate through the branch [kg/s].
    """
    def __init__(self,
                 name: str,
                 mass_flow: float,  # kg/s
                 ):
        
        self.name = name
        self.mdot = mass_flow

    def __str__(self):
        return f"Branch (name={self.name}, mdot={self.mdot:.3e} kg/s)"




# ----- Volume Classes -----
class Tank(Volume):
    """
    Tank model representing a pressurized fluid reservoir.

    A Tank is a control volume with a predefined propellant.
    
    Parameters
    ----------
    name : str
        Identifier for the tank.
    pressure : float
        Tank pressure [Pa].
    volume : float
        Tank volume [m^3].
    density : float
        Fluid density within the tank [kg/m^3].
    """
    def __init__(self, name, pressure, volume, density: float):
        super().__init__(name, pressure, volume)
        self.rho = density  # kg/m^3

    def __str__(self):
        return (
            f"Tank (name={self.name}, p={self.p:.3e} Pa, "
            f"V={self.V:.3e} m^3, rho={self.rho:.3e} kg/m^3)"
        )
    

class InjectorManifold(Volume):
    """
    Injector Manifold model representing a pressurized injector manifold.

    Contains predefined propellant.
    
    Parameters
    ----------
    name : str
        Identifier for the tank.
    pressure : float
        Tank pressure [Pa].
    volume : float
        Tank volume [m^3].
    density : float
        Fluid density within the tank [kg/m^3].
    """
    def __init__(self, name, pressure, volume, density: float):
        super().__init__(name, pressure, volume)
        self.rho = density  # kg/m^3

    def __str__(self):
        return (
            f"Injector Manifold (name={self.name}, p={self.p:.3e} Pa, "
            f"V={self.V:.3e} m^3, rho={self.rho:.3e} kg/m^3)"
        )

class CombustionChamber(Volume):
    """
    TCA Combustion Chamber

    Represents a combustion chamber characterized entirely by the chamber
    pressure and mixture ratio.
    
    Parameters
    ----------
    name : str
        Identifier for the combustion chamber.
    pressure : float
        Chamber pressure [Pa].
    volume : float
        Chamber volume [m^3].
    mixture_ratio : float
        Oxidizer-to-fuel mass ratio (O/F).
    """
    def __init__(self, name, pressure, volume, mixture_ratio: float):
        super().__init__(name, pressure, volume)
        self.MR = mixture_ratio

    def __str__(self):
        return (
            f"Combustion Chamber (name={self.name}, p={self.p:.3e} Pa, "
            f"V={self.V:.3e} m^3, MR={self.MR:.3f})"
        )


# ----- Branch Classes -----
class Orifice(Branch):
    """
    Orifice component contains a flow area attribute.

    An Orifice is used to model flow devices with known CdA
    
    Parameters
    ----------
    name : str
        Identifier for the valve.
    mass_flow : float
        Mass flow rate through the valve [kg/s].
    CdA : float
        Effective discharge area [m^2].
    """
    def __init__(self, name, mass_flow, CdA: float):
        super().__init__(name, mass_flow)
        self.CdA = CdA  # m^2

    def __str__(self):
        return (
            f"Orifice (name={self.name}, mdot={self.mdot:.3e} kg/s, "
            f"CdA={self.CdA:.3e} m^2)"
        )
class Line(Branch):
    """
    Represents an incompressible flow device

    A Line carries a certain flow resistance, given by CdA, and also
    a certain inertance due to te line length and area.
    
    Parameters
    ----------
    name : str
        Identifier for the line.
    mass_flow : float
        Mass flow rate through the line [kg/s].
    length : float
        Line length [m].
    cross_sectional_area : float
        Internal flow area [m^2].
    Cd : float
        Discharge coefficient (dimensionless, 0 <= Cd <= 1).
    """
    def __init__(self, name, mass_flow, length: float, cross_sectional_area: float, Cd: float):
        super().__init__(name, mass_flow)
        self.L = length
        self.A = cross_sectional_area   # m^2
        self.Cd = Cd
        self.CdA = self.A * self.Cd

    def __str__(self):
        return (
            f"Line (name={self.name}, mdot={self.mdot:.3e} kg/s, "
            f"L={self.L:.3e} m, A={self.A:.3e} m^2, Cd={self.Cd:.3f})"
        )


class Valve(Branch):
    """
    Valve component contains a variable flow area attribute.

    A Valve can be used a simple open/close flow controller or
    as a throttlable flow adjuster.
    
    Parameters
    ----------
    name : str
        Identifier for the valve.
    mass_flow : float
        Mass flow rate through the valve [kg/s].
    CdA : float
        Effective discharge area [m^2].
    """
    def __init__(self, name, mass_flow, CdA: float):
        super().__init__(name, mass_flow)
        self.CdA = CdA  # m^2

    def __str__(self):
        return (
            f"Valve (name={self.name}, mdot={self.mdot:.3e} kg/s, "
            f"CdA={self.CdA:.3e} m^2)"
        )
    

class Nozzle(Branch):
    """
    Nozzle uses CEA to 

    A Valve can be used a simple open/close flow controller or
    as a throttlable flow adjuster.
    
    Parameters
    ----------
    name : str
        Identifier for the nozzle.
    mass_flow : float
        Mass flow rate through the nozzle [kg/s].
    throat_area : float
        Throat area [m^2].
    expansion_ratio : float
        (Exit Area) / (Throat Area).
    contraction_ratio: float
        (Injector Face Area) / (Throat Area).
    nfz : int
        Number frozen zone (0: Equilibrium, 1: Fully frozen, 2: Frozen from throat).
    """
    def __init__(self, name, mass_flow, throat_area: float, expansion_ratio: float, contraction_ratio: float, nfz: int):
        super().__init__(name, mass_flow)
        self.At = throat_area
        self.eps = expansion_ratio
        self.eps_c = contraction_ratio
        self.nfz = nfz

    def __str__(self):
        return (
            f"Nozzle (name={self.name}, mdot={self.mdot:.3e} kg/s, At={self.CdA:.3e} m^2), "
            f"eps={self.eps:.3e}, eps_c={self.eps_c:.3e}, nfz={self.nfz:.3e}"
        )