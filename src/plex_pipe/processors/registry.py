from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Type

from pydantic import BaseModel

from plex_pipe.processors.base import BaseOp


@dataclass
class RegistryEntry:
    processor_class: Type[BaseOp]
    param_model: Type[BaseModel]


Kind = Literal[
    "mask_builder",
    "object_segmenter",
    "image_transformer",
]

REGISTRY: Dict[Kind, Dict[str, Type[BaseOp]]] = {
    "mask_builder": {},
    "object_segmenter": {},
    "image_transformer": {},
}


def register(kind: Kind, name: str):
    """
    Decorator for plugin registration that also captures the Params model.
    """

    def deco(cls: Type[BaseOp]):
        # Get the parameter model from the class, defaulting to an empty one
        param_model = getattr(cls, "Params", BaseModel)

        # Create and store the complete registry entry
        REGISTRY[kind][name] = RegistryEntry(
            processor_class=cls, param_model=param_model
        )
        cls.kind = kind
        cls.type_name = name
        return cls

    return deco


def build_processor(kind: Kind, name: str, **cfg: Any) -> BaseOp:
    """Instantiate an operation by kind and name."""
    if kind not in REGISTRY:
        available_kinds = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown kind '{kind}'. Available kinds: {available_kinds or '(none)'}"
        )

    registry_for_kind = REGISTRY[kind]
    if name not in registry_for_kind:
        available_names = ", ".join(sorted(registry_for_kind))
        raise ValueError(
            f"Unknown {kind} '{name}'. Available: {available_names or '(none)'}"
        )

    cls = registry_for_kind[name].processor_class
    obj = cls(**cfg)

    return obj
