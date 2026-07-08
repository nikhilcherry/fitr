"""fitr: analysis-by-synthesis model fitting for phase-folded light curves."""

from .compare import Comparison, compare
from .fit import FitResult, fit_all, fit_model
from .fold import bin_folded, fold
from .io import FitrInputError, LightCurve, load_lightcurve
from .report import to_json, to_text

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LightCurve",
    "FitrInputError",
    "load_lightcurve",
    "fold",
    "bin_folded",
    "FitResult",
    "fit_model",
    "fit_all",
    "Comparison",
    "compare",
    "to_json",
    "to_text",
]
