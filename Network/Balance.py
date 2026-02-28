from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from TestStand import TestStand


# Helper Functions:

def _split_path(path: str) -> tuple[str, str]:
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(
            f"Expected path format 'Object.attr', got {path!r}"
        )
    return parts[0], parts[1]


def _make_tune_set(path: str) -> Callable[["TestStand", float], None]:
    obj_name, attr = _split_path(path)

    def _set(ts: "TestStand", value: float) -> None:
        setattr(getattr(ts, obj_name), attr, float(value))

    return _set


def _make_measure_attr(path: str) -> Callable[["TestStand"], float]:
    obj_name, attr = _split_path(path)

    def _get(ts: "TestStand") -> float:
        return float(getattr(getattr(ts, obj_name), attr))

    return _get



# Balance class

@dataclass(frozen=True)
class Balance:
    """
    Beginner-friendly balance object (no shorthand keywords).

    Rules:
    - tune must be "Object.attr"  (e.g., "OxThrottleValve.CdA", "TCA.At")
    - measure must be "Object.attr" (e.g., "MainChamber.p", "MainChamber.MR", "TCA.F")
      OR an advanced callable(solved_ts)->float.

    Example:
        Balance(
            tune="OxThrottleValve.CdA",
            measure="MainChamber.MR",
            target=2.0,
            bounds=(1e-10, 1e-4),
            tol=1e-5,
        )
    """

    tune: str
    measure: Union[str, Callable[["TestStand"], float]]
    target: float
    bounds: Tuple[float, float] = (1e-8, 5e-4)
    tol: float = 1e-6
    name: str | None = None

    def __post_init__(self):
        lo, hi = self.bounds
        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError("bounds must be (lo, hi) with 0 < lo < hi.")

        # Build tune_set
        tune_set = _make_tune_set(self.tune)

        # Build measure function
        if callable(self.measure):
            measure_fn = self.measure
            measure_label = getattr(self.measure, "__name__", "custom")
        else:
            m = self.measure.strip()
            if "." not in m:
                raise ValueError(
                    f"measure must be an 'Object.attr' path (e.g., 'MainChamber.p'), got {m!r}"
                )
            measure_fn = _make_measure_attr(m)
            measure_label = m

        # Auto-generate name if not provided
        if self.name is None:
            auto_name = f"Tune {self.tune} until {measure_label} = {self.target}"
            object.__setattr__(self, "name", auto_name)

        # Store compiled versions (private attributes)
        object.__setattr__(self, "_tune_set", tune_set)
        object.__setattr__(self, "_measure_fn", measure_fn)

    @property
    def tune_set(self):
        return self._tune_set

    @property
    def measure_fn(self):
        return self._measure_fn

    def describe(self) -> str:
        lo, hi = self.bounds
        return (
            f"Balance[{self.name}] "
            f"target={self.target} "
            f"knob_bounds=[{lo:.3e}, {hi:.3e}] "
            f"tol={self.tol}"
        )