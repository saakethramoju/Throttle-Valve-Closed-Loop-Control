from pyfluids import FluidsList as pyFluidsList
from pyfluids import Fluid as pyFluid
from pyfluids import Mixture as pyMixture
from pyfluids import Input as pyInput
from scipy.optimize import root_scalar
from typing import List, Union, Dict, Tuple
import numpy as np


class Fluid:
    """
    High-level wrapper around pyfluids Fluid/Mixture objects with added
    convenience methods for initialization, state updates, and property access.

    Supports both pure fluids and mixtures (by mole or mass fractions).
    """

    _ALIASES = {
        "rp-1": "nDodecane",
        "rp1": "nDodecane",
        "jet-a": "nDodecane",
        "jeta": "nDodecane",
        "kerosene": "nDodecane",
    }

    def __init__(
        self,
        fluid: Union[str, Dict[str, float]],
        basis: str = "mass",
        P: float = None,
        h: float = None,
        T: float = None,
        Q: float = None,
    ):
        """
        Initialize a Fluid state.

        Parameters
        ----------
        fluid : str or dict
            Pure fluid name (str) or dictionary of {fluid_name: fraction}.
        basis : str
            "mole" or "mass" basis for mixture fractions (default "mass").
        P : float, optional
            Absolute pressure in Pa.
        h : float, optional
            Specific enthalpy in J/kg.
        T : float, optional
            Absolute temperature in K.
        Q : float, optional
            Vapor quality (0-1).
        """
        valid_fluids = Fluid.get_available_fluids()

        self._fluids: List[str] = []          # backend names (e.g., nDodecane)
        self._display_names: List[str] = []   # user-facing names (e.g., RP-1)

        # -------------------------------
        # Handle pure fluid input
        # -------------------------------
        if isinstance(fluid, str):
            backend, display = Fluid._normalize_name(fluid)
            if backend not in valid_fluids:
                raise ValueError(
                    f"Invalid fluid '{fluid}'. "
                    f"Use Fluid.show_available_fluids() to check valid names."
                )
            self._fluids = [backend]
            self._display_names = [display]
            self._mole_fractions = np.array([1.0])
            self._mass_fractions = np.array([1.0])
            self._mixture = False

        # -------------------------------
        # Handle mixture input
        # -------------------------------
        elif isinstance(fluid, dict):
            # ---- Single-component dict case ----
            if len(fluid) == 1:
                f, frac = next(iter(fluid.items()))
                if not np.isclose(frac, 1.0, atol=1e-12):
                    raise ValueError(f"Single-component dict must have fraction = 1.0, got {frac}")
                backend, display = Fluid._normalize_name(f)
                if backend not in valid_fluids:
                    raise ValueError(
                        f"Invalid fluid '{f}'. "
                        f"Use Fluid.show_available_fluids() to check valid names."
                    )
                self._fluids = [backend]
                self._display_names = [display]
                self._mole_fractions = np.array([1.0])
                self._mass_fractions = np.array([1.0])
                self._mixture = False

            # ---- True mixture ----
            else:
                tmp: Dict[str, Tuple[float, List[str]]] = {}
                for user_name, frac in fluid.items():
                    backend, display = Fluid._normalize_name(user_name)
                    if backend not in valid_fluids:
                        raise ValueError(
                            f"Invalid fluid '{user_name}' (backend '{backend}' not found). "
                            f"Use Fluid.show_available_fluids() to check valid names."
                        )
                    total, names = tmp.get(backend, (0.0, []))
                    tmp[backend] = (total + float(frac), names + [display])

                self._fluids = list(tmp.keys())
                fractions = np.array([v[0] for v in tmp.values()], dtype=float)
                self._display_names = [
                    ", ".join(sorted(set(names))) for (_, names) in tmp.values()
                ]

                if basis == "mole":
                    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                        raise ValueError("Mole fractions must sum to 1.0")
                    self._mole_fractions = fractions
                    self._mass_fractions = Fluid.mole_to_mass(self._fluids, fractions)
                elif basis == "mass":
                    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
                        raise ValueError("Mass fractions must sum to 1.0")
                    self._mass_fractions = fractions
                    self._mole_fractions = Fluid.mass_to_mole(self._fluids, fractions)
                else:
                    raise ValueError("basis must be 'mole' or 'mass'")
                self._mixture = len(self._fluids) > 1

        else:
            raise TypeError("fluid must be a string (pure) or dict (mixture)")

        # -------------------------------
        # Build pyfluids backend
        # -------------------------------
        self._backend = (
            pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
            if self._mixture else pyFluid(pyFluidsList[self._fluids[0]])
        )

        # -------------------------------
        # Initialize state
        # -------------------------------
        if P is not None and h is not None:
            self._P, self._h = P, h
        elif P is not None and T is not None:
            self._P = P
            self._h = self._backend.with_state(pyInput.temperature(T), pyInput.pressure(P)).enthalpy
        elif P is not None and Q is not None:
            self._P = P
            self._h = self._backend.with_state(pyInput.quality(Q), pyInput.pressure(P)).enthalpy
        elif T is not None and Q is not None:
            st = self._backend.with_state(pyInput.quality(Q), pyInput.temperature(T))
            self._P, self._h = st.pressure, st.enthalpy
        else:
            raise LookupError("Please provide at least two thermodynamic properties!")

        self.set_pyfluid()


    # ---------------- Core ---------------- #
    def set_pyfluid(self):
        """Rebuild backend pyfluid state using current P and h."""
        if self._mixture:
            T, Q = Fluid.get_temperature_and_quality(self._backend, self._P, self._h)
            if 0.0 < Q < 1.0:  # inside dome
                self._pyfluid = self._backend.with_state(pyInput.quality(Q), pyInput.pressure(self._P))
            else:  # single-phase
                self._pyfluid = self._backend.with_state(pyInput.temperature(T), pyInput.pressure(self._P))
        else:
            self._pyfluid = self._backend.with_state(pyInput.pressure(self._P), pyInput.enthalpy(self._h))

    # ---------------- Fractions ---------------- #
    @property
    def mole_fractions(self) -> dict:
        """Return mole fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mole_fractions)}

    @mole_fractions.setter
    def mole_fractions(self, value: List[float]):
        """Update mole fractions (must sum to 1)."""
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mole fractions for a pure fluid")
        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mole fractions must sum to 1.0")
        self._mole_fractions = np.array(value, dtype=float)
        self._mass_fractions = Fluid.mole_to_mass(self._fluids, value)
        self._backend = pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
        if self._P is not None and self._h is not None:
            self.set_pyfluid()

    @property
    def mass_fractions(self) -> dict:
        """Return mass fractions as {fluid_name: value}."""
        return {f: float(x) for f, x in zip(self._fluids, self._mass_fractions)}

    @mass_fractions.setter
    def mass_fractions(self, value: List[float]):
        """Update mass fractions (must sum to 1)."""
        if len(self._fluids) == 1:
            raise ValueError("Cannot change mass fractions for a pure fluid")
        if not np.isclose(sum(value), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1.0")
        self._mass_fractions = np.array(value, dtype=float)
        self._mole_fractions = Fluid.mass_to_mole(self._fluids, value)
        self._backend = pyMixture([pyFluidsList[f] for f in self._fluids], self._mass_fractions)
        if self._P is not None and self._h is not None:
            self.set_pyfluid()

    # ---------------- State setters ---------------- #
    @property
    def pressure(self) -> float:
        """Absolute pressure (Pa)."""
        return self._P

    @pressure.setter
    def pressure(self, value: float):
        self._P = value
        if self._h is not None:
            self.set_pyfluid()

    @property
    def enthalpy(self) -> float:
        """Mass-specific enthalpy (J/kg)."""
        return self._h

    @enthalpy.setter
    def enthalpy(self, value: float):
        self._h = value
        if self._P is not None:
            self.set_pyfluid()

    @property
    def HP(self) -> Tuple[float, float]:
        """Return (P [Pa], h [J/kg])."""
        return self._P, self._h

    @HP.setter
    def HP(self, values: Tuple[float, float]):
        """Update pressure and enthalpy simultaneously."""
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("HP must be set with (P, h)")
        self._P, self._h = values
        self.set_pyfluid()

    # ---------------- Thermo properties ---------------- #
    @property
    def species(self) -> List[str]:
        return self._display_names
    
    @property
    def temperature(self) -> float:
        """Absolute temperature (K)."""
        return self._pyfluid.temperature

    @property
    def phase(self) -> str:
        """
        Thermodynamic phase name as returned by CoolProp.

        Possible values:
        - "Unknown"           : Phase could not be determined
        - "Liquid"            : Single-phase liquid
        - "Supercritical"     : Supercritical state (above Tc, Pc)
        - "SupercriticalGas"  : Supercritical but more gas-like
        - "SupercriticalLiquid": Supercritical but more liquid-like
        - "Gas"               : Single-phase vapor/gas
        - "TwoPhase"          : Saturated mixture of liquid + vapor

        Returns
        -------
        str
            Phase string reported by CoolProp backend.
        """
        return self._pyfluid.phase.name


    @property
    def compressibility(self) -> float:
        """Compressibility factor Z = pV/RT (dimensionless)."""
        return self._pyfluid.compressibility

    @property
    def conductivity(self) -> float:
        """Thermal conductivity (W/m·K)."""
        return self._pyfluid.conductivity

    @property
    def critical_pressure(self) -> float:
        """Critical point pressure (Pa)."""
        return self._pyfluid.critical_pressure

    @property
    def critical_temperature(self) -> float:
        """Critical point temperature (K)."""
        return self._pyfluid.critical_temperature

    @property
    def density(self) -> float:
        """Mass density (kg/m³)."""
        return self._pyfluid.density

    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity (Pa·s)."""
        return self._pyfluid.dynamic_viscosity

    @property
    def entropy(self) -> float:
        """Mass-specific entropy (J/kg·K)."""
        return self._pyfluid.entropy

    @property
    def freezing_temperature(self) -> float:
        """Freezing point temperature (K)."""
        return self._pyfluid.freezing_temperature

    @property
    def internal_energy(self) -> float:
        """Mass-specific internal energy (J/kg)."""
        return self._pyfluid.internal_energy

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity (m²/s)."""
        return self._pyfluid.kinematic_viscosity

    @property
    def maximum_pressure(self) -> float:
        """Maximum valid pressure for backend model (Pa)."""
        return self._pyfluid.max_pressure

    @property
    def maximum_temperature(self) -> float:
        """Maximum valid temperature for backend model (K)."""
        return self._pyfluid.max_temperature

    @property
    def minimum_pressure(self) -> float:
        """Minimum valid pressure for backend model (Pa)."""
        return self._pyfluid.min_pressure

    @property
    def minimum_temperature(self) -> float:
        """Minimum valid temperature for backend model (K)."""
        return self._pyfluid.min_temperature

    @property
    def molar_mass(self) -> float:
        """Molar mass (kg/mol)."""
        return self._pyfluid.molar_mass

    @property
    def prandtl(self) -> float:
        """Prandtl number Pr = μCp/k (dimensionless)."""
        return self._pyfluid.prandtl

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound (m/s)."""
        return self._pyfluid.sound_speed

    @property
    def specific_heat(self) -> float:
        """Mass-specific heat capacity at constant pressure Cp (J/kg·K)."""
        return self._pyfluid.specific_heat

    @property
    def specific_volume(self) -> float:
        """Specific volume (m³/kg)."""
        return self._pyfluid.specific_volume

    @property
    def surface_tension(self) -> float:
        """Surface tension (N/m)."""
        return self._pyfluid.surface_tension

    @property
    def triple_pressure(self) -> float:
        """Triple point pressure (Pa)."""
        return self._pyfluid.triple_pressure

    @property
    def triple_temperature(self) -> float:
        """Triple point temperature (K)."""
        return self._pyfluid.triple_temperature

    @property
    def is_mixture(self) -> bool:
        """Return True if this fluid is a mixture, False if pure."""
        return self._mixture


    @property
    def quality(self) -> float:
        """
        Vapor quality (0-1). Only meaningful in TwoPhase region.
        Returns 0.0 for liquid-like single phase, 1.0 for gas/supercritical single phase.
        """
        ph = self.phase  # uses self._pyfluid.phase.name (no PQ flash)

        if ph == "TwoPhase":
            # In two-phase, CoolProp/pyfluids quality should be valid
            return float(self._pyfluid.quality)

        # Outside two-phase, "quality" isn't defined.
        # Pick a consistent convention:
        if ph in ("Gas", "Supercritical", "SupercriticalGas"):
            return 1.0
        if ph in ("Liquid", "SupercriticalLiquid"):
            return 0.0

        # Unknown -> don't crash; just return NaN
        return float("nan")

    @property
    def saturation_temperature(self) -> float:
        """Saturation temperature only defined for P <= Pc."""
        if self.pressure > self.critical_pressure:
            return None
        return self._backend.with_state(pyInput.pressure(self._P), pyInput.quality(1.0)).temperature

    # ---------------- String output ---------------- #
    def _safe(self, value, fmt=".3e"):
        if value is None: return "N/A"
        try: return f"{value:{fmt}}"
        except Exception: return str(value)

    def __str__(self):
        def format_dict(d: dict, decimals=3):
            return {k: round(v, decimals) for k, v in d.items()}
        rows = [
            ("Fluid(s)", ", ".join(self._display_names)),
            ("Mole fractions", format_dict(self.mole_fractions, 3)),
            ("Mass fractions", format_dict(self.mass_fractions, 3)),
            ("Phase", self.phase),
            ("Pressure [Pa]", self._safe(self.pressure, ".3e")),
            ("Temperature [K]", self._safe(self.temperature, ".2f")),
            ("Density [kg/m³]", self._safe(self.density, ".3f")),
            ("Quality", self._safe(self.quality, ".3f")),
            ("Internal energy [J/kg]", self._safe(self.internal_energy, ".3e")),
            ("Enthalpy [J/kg]", self._safe(self.enthalpy, ".3e")),
            ("Entropy [J/kg-K]", self._safe(self.entropy, ".3e")),
            ("Dynamic viscosity [Pa·s]", self._safe(self.dynamic_viscosity, ".3e")),
            ("Conductivity [W/m-K]", self._safe(self.conductivity, ".3f")),
            ("Saturation temperature [K]", self._safe(self.saturation_temperature, ".2f")),
            ("Molar mass [kg/mol]", self._safe(self.molar_mass, ".6f")),
        ]
        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{key:<{width}} : {val}" for key, val in rows)
    
    def __repr__(self) -> str:
        """
        Representation of the Fluid object.

        Includes class name, species, pressure, enthalpy, and temperature.
        Useful for debugging, logging, or interactive sessions.

        Returns
        -------
        str
            String in the format:
            Fluid(species=['Nitrogen', 'Oxygen'], P=101325 Pa, h=2.98e+05 J/kg, T=298.15 K)
        """
        species_str = ", ".join(self._display_names)
        return (f"{self.__class__.__name__}(species=[{species_str}], "
                f"P={self._P:.3e} Pa, h={self._h:.3e} J/kg, T={self.temperature:.2f} K)")

    # ---------------- Utilities ---------------- #

    @staticmethod
    def _normalize_name(user_name: str) -> Tuple[str, str]:
        """
        Return (backend_name, display_name). If user_name is an alias,
        map to nDodecane for backend but keep user_name for display.
        """
        display = user_name  # keep exactly what the user typed
        key = user_name.strip().lower()
        key = key.replace(" ", "")    # remove spaces
        key = key.replace("_", "-")   # unify underscores to dashes
        backend = Fluid._ALIASES.get(key, user_name)
        return backend, display

    

    @staticmethod
    def mole_to_mass(fluids: List[str], mole_fractions: List[float]):
        """Convert mole fractions → mass fractions."""
        if not np.isclose(sum(mole_fractions), 1.0, atol=1e-6): raise ValueError("Mole fractions must sum to 1.0")
        molar_masses = np.array([pyFluid(pyFluidsList[f]).molar_mass for f in fluids])
        m_bar = np.dot(mole_fractions, molar_masses)
        return np.asarray(mole_fractions) * molar_masses / m_bar

    @staticmethod
    def mass_to_mole(fluids: List[str], mass_fractions: List[float]):
        """Convert mass fractions → mole fractions."""
        if not np.isclose(sum(mass_fractions), 1.0, atol=1e-6): raise ValueError("Mole fractions must sum to 1.0")
        molar_masses = np.array([pyFluid(pyFluidsList[f]).molar_mass for f in fluids])
        inv = np.asarray(mass_fractions) / molar_masses
        return inv / inv.sum()

    @staticmethod
    def get_temperature_and_quality(fluid: pyFluid, P: float, target_enthalpy: float) -> Tuple[float, float]:
        """
        Given a backend fluid, pressure, and enthalpy, return (T, Q).
        """
        h_liquid = fluid.with_state(pyInput.quality(0), pyInput.pressure(P)).enthalpy
        h_vapor = fluid.with_state(pyInput.quality(1), pyInput.pressure(P)).enthalpy
        h = target_enthalpy
        if h_liquid <= h <= h_vapor:
            Q = (h - h_liquid) / (h_vapor - h_liquid)
            T = fluid.with_state(pyInput.quality(Q), pyInput.pressure(P)).temperature
        else:
            def residual(T):
                try: return fluid.with_state(pyInput.temperature(T), pyInput.pressure(P)).enthalpy - h
                except: return 1e13
            sol = root_scalar(residual, method="brentq", bracket=[fluid.min_temperature, fluid.max_temperature])
            T = sol.root
            Q = 0.0 if h < h_liquid else 1.0
        return T, Q

    @staticmethod
    def show_available_fluids():
        """Print and return available fluid names."""
        for f in pyFluidsList:
            if f.pure and f.name: print(f.name)
        return [f.name for f in pyFluidsList if f.pure and f.name]

    @staticmethod
    def get_available_fluids():
        """Return available fluid names."""
        return [f.name for f in pyFluidsList if f.pure and f.name]



if __name__ == "__main__":


    f = Fluid({"Nitrogen": 0.78, "Oxygen": 0.21, "Argon": 0.01}, basis="mole", P=101325, T=298.15)
    #f = Fluid({"Nitrogen": 1}, P=101325, h=311200)
    #f = Fluid("Methane", P=3e6, Q=0.1)
    f = Fluid("nDodecane", P=101325, T=300)
    print(f)
    print(f.minimum_pressure)
    #f.HP = 3.1e5, 2e5
    #f.mass_fractions = [0.4, 0.3, 0.3]
    print("-------------------")
    f = Fluid("Air", P=101325, T=298.15)
    print(f)
    #Fluid.show_available_fluids()
    #print(Fluid.get_saturation_pressure({"Nitrogen": 0.79, "Oxygen": 0.11, "Methane": 0.1}, T=120))
