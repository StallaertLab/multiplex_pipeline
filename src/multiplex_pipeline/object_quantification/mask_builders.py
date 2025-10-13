from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Sequence, Type
import numpy as np


################################################################################
# Registry
################################################################################
BUILDER_REGISTRY: Dict[str, Type["BaseBuilder"]] = {}


def register_builder(mask_type: str):
    """
    Decorator that registers a builder class under a given name.
    Usage:
        @register_builder("ring")
        class RingBuilder(BaseBuilder): ...
    """
    def _wrap(cls: Type["BaseBuilder"]):
        BUILDER_REGISTRY[mask_type] = cls
        cls.mask_type = mask_type
        return cls
    return _wrap


################################################################################
# Base class
################################################################################
class BaseBuilder(ABC):
    """
    Abstract base class for all derived-mask builders.
    Subclasses define algorithm-specific logic in `build(...)`
    and may override constructor validation.
    """

    # -------- Validation (request-level) -------- #
    @abstractmethod
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate config for a given builder."""
        ...


    # -------- Algorithm (subclass responsibility) -------- #
    @abstractmethod
    def run(self, *, sdata: Any) -> np.ndarray:
        """
        Core algorithm: compute a derived mask from one or more sources.
        Must return a NumPy array representing the new label image.
        """
        ...

    # -------- Factory for lookup & instantiation -------- #
    @classmethod
    def create(cls, mask_type: str, cfg) -> "BaseBuilder":
        """
        Factory method: instantiate a builder by its registered mask_type.
        Raises ValueError if not found.
        """
        try:
            BuilderClass = BUILDER_REGISTRY[mask_type]
        except KeyError as e:
            available = ", ".join(sorted(BUILDER_REGISTRY))
            raise ValueError(f"Unknown mask_type '{mask_type}'. "
                             f"Available: {available or '(none)'}") from e
        return BuilderClass(cfg)

    # -------- Helper: list all registered types -------- #
    @classmethod
    def available(cls) -> list[str]:
        """List all registered builder types."""
        return sorted(BUILDER_REGISTRY.keys())

################################################################################
# Specific builders
################################################################################

@register_builder("subtraction")
class SubtractionBuilder(BaseBuilder):

    def __init__(self, cfg):
        # Algorithm-specific validation
        self.validate_config(cfg)
        
    def validate_config(self, cfg):
        if cfg:
            raise ValueError("SubtractionBuilder takes no parameters.")
    
    def run(self, mask_cell, mask_nucleus):
        if mask_cell.shape != mask_nucleus.shape:
            raise ValueError("Source masks must have the same shape for subtraction.")
        result = mask_cell.copy()
        result[mask_nucleus > 0] = 0  # zero out regions where mask_nucleus is present
        return result
    
@register_builder("ring")
class RingBuilder(BaseBuilder):

    def __init__(self, cfg):
        # Algorithm-specific validation
        self.validate_config(cfg)
        self.outer = cfg["outer"]
        self.inner = cfg["inner"] 
        
    def validate_config(self, cfg):
        if not isinstance(cfg, dict):
            raise ValueError("RingBuilder configuration must be a dictionary.")
        if "outer" in cfg and (not isinstance(cfg["outer"], int) or cfg["outer"] <= 0):
            raise ValueError("'outer' radius must be a positive integer")
        if "inner" in cfg and (not isinstance(cfg["inner"], int) or cfg["inner"] < 0):
            raise ValueError("'inner' radius must be a non-negative integer.")
    
    def run(self, mask):
        from skimage.segmentation import expand_labels
        mask_big = expand_labels(mask, self.outer)
        mask_small = expand_labels(mask, self.inner)
        result = mask_big - mask_small
        return result