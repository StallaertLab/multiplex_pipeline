from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    List,
    Mapping,
    Sequence,
    Tuple,
)


class OutputType(str, Enum):
    """Allowed output types for processors."""

    IMAGE = "image"
    LABELS = "labels"


class BaseOp(ABC):
    """Common parent for all operations."""

    kind: str
    type_name: str
    EXPECTED_INPUTS: int | None = 1
    EXPECTED_OUTPUTS: int | None = 1
    OUTPUT_TYPE: OutputType

    def __init__(self, **cfg: Any):
        self.cfg = dict(cfg)

        if not isinstance(self.OUTPUT_TYPE, OutputType):
            raise TypeError(
                f"{self.__class__.__name__}.OUTPUT_TYPE must be an OutputType enum, "
                f"got {self.OUTPUT_TYPE!r}"
            )

    @staticmethod
    def _normalize_names(
        names: str | Sequence[str] | None, label: str
    ) -> List[str]:
        """Return a list of strings; accept 'name' or ['name'] (None -> [])."""
        if names is None:
            return []
        if isinstance(names, str):
            return [names]
        if isinstance(names, Sequence) and all(
            isinstance(x, str) for x in names
        ):
            return list(names)
        raise TypeError(
            f"{label} must be a string or a sequence of strings, got {type(names).__name__}."
        )

    def validate_io(
        self,
        inputs: str | Sequence[str] | None,
        outputs: str | Sequence[str] | None,
    ) -> Tuple[List[str], List[str]]:
        """
        Normalize and validate I/O names against this class's declared arity.
        Returns (input_names, output_names) as lists of strings.
        """
        in_list = self._normalize_names(inputs, "inputs")
        out_list = self._normalize_names(outputs, "outputs")

        if (
            self.EXPECTED_INPUTS is not None
            and len(in_list) != self.EXPECTED_INPUTS
        ):
            raise ValueError(
                f"{self.__class__.__name__}: expected {self.EXPECTED_INPUTS} input name(s), "
                f"got {len(in_list)}: {in_list!r}"
            )
        if (
            self.EXPECTED_OUTPUTS is not None
            and len(out_list) != self.EXPECTED_OUTPUTS
        ):
            raise ValueError(
                f"{self.__class__.__name__}: expected {self.EXPECTED_OUTPUTS} output name(s), "
                f"got {len(out_list)}: {out_list!r}"
            )

        return in_list, out_list

    def initialize(self):
        """Optional hook for subclass-specific setup."""
        return None

    @abstractmethod
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate configuration. Raise ValueError with clear message if invalid."""
        ...

    @abstractmethod
    def run(self, *sources: Any) -> Any:
        """Perform the operation on given sources."""
        ...

    def __repr__(self) -> str:
        """Unambiguous developer-oriented representation."""
        return (
            f"<{self.__class__.__name__}("
            f"kind='{getattr(self, 'kind', '?')}', "
            f"type='{getattr(self, 'type_name', '?')}', "
            f"cfg={self.cfg})>"
        )

    def __str__(self) -> str:
        """Human-friendly printout, e.g. in logs."""
        return f"{self.kind}:{self.type_name} {self.cfg}"
