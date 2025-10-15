from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Literal, Mapping, Optional
from multiplex_pipeline.processors.base import BaseOp

Kind = Literal["mask_builder", "mask_cleaner", "object_segmenter", "image_mixer", "image_transformer"]

REGISTRY: Dict[Kind, Dict[str, Type["BaseOp"]]] = {
    "mask_builder": {},
    "mask_cleaner": {},
    "object_segmenter": {},
    "image_mixer": {},
    "image_transformer": {},
}

def register(kind: Kind, name: str):
    """Decorator for plugin registration."""
    def deco(cls: Type["BaseOp"]):
        REGISTRY[kind][name] = cls
        cls.kind = kind
        cls.type_name = name
        return cls
    return deco

def build_processor(kind: Kind, name: str, **cfg: Any) -> "BaseOp":
    """Instantiate an operation by kind and name."""
    if kind not in REGISTRY:
        available_kinds = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown kind '{kind}'. "
            f"Available kinds: {available_kinds or '(none)'}"
        )

    registry_for_kind = REGISTRY[kind]
    if name not in registry_for_kind:
        available_names = ", ".join(sorted(registry_for_kind))
        raise ValueError(
            f"Unknown {kind} '{name}'. "
            f"Available: {available_names or '(none)'}"
        )

    cls = registry_for_kind[name]
    obj = cls(**cfg)
    obj.validate_config(cfg)
    obj.initialize()
    
    return obj