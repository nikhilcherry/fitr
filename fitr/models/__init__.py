"""Model registry. Adding a fifth model requires only adding it here."""

from __future__ import annotations

from .base import Model
from .blend import BlendModel
from .eb import EBModel
from .planet import PlanetModel
from .starspot import StarspotModel

ALL_MODELS: dict[str, Model] = {
    "planet": PlanetModel(),
    "eb": EBModel(),
    "blend": BlendModel(),
    "starspot": StarspotModel(),
}

__all__ = ["Model", "ALL_MODELS", "PlanetModel", "EBModel", "BlendModel", "StarspotModel"]
