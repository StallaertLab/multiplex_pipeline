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
    """Enumerates allowed output artifact types for operations.

    Values:
        IMAGE: Image-like array data (e.g., numpy arrays).
        LABELS: Label/segmentation data.
    """

    IMAGE = "image"
    LABELS = "labels"


class BaseOp(ABC):
    """An abstract base class for all processing operations.

    This class provides a common interface for all operations, ensuring that they can be called by a controller in a
    consistent manner. Subclasses must implement the `validate_config` and
    `run` methods.

    Attributes:
        kind: A string that identifies the kind of operation ("image_transformer",
            "mask_builder" or "object_segmenter"). Assigned by the registry.
        type_name: A string that specifies the type of the operation (e.g.,
            "normalize", "instanseg"). Assigned by the registry.
        EXPECTED_INPUTS: The number of expected inputs for the operation. If
            None, a variable number of inputs is allowed.
        EXPECTED_OUTPUTS: The number of expected outputs for the operation. If
            None, a variable number of outputs is allowed.
        OUTPUT_TYPE: The type of output produced by the operation.
        cfg: A dictionary containing the configuration for the operation.
    """

    kind: str
    type_name: str
    EXPECTED_INPUTS: int | None = 1
    EXPECTED_OUTPUTS: int | None = 1
    OUTPUT_TYPE: OutputType

    def __init__(self, **cfg: Any):
        """Initialize the operation with keyword configuration.

        Args:
            **cfg: Free-form configuration for the operation.

        Raises:
            TypeError: If `OUTPUT_TYPE` is not an `OutputType` enum member.
        """
        self.cfg = dict(cfg)

        if not isinstance(self.OUTPUT_TYPE, OutputType):
            raise TypeError(
                f"{self.__class__.__name__}.OUTPUT_TYPE must be an OutputType enum, "
                f"got {self.OUTPUT_TYPE!r}"
            )

    @staticmethod
    def _normalize_names(names: str | Sequence[str] | None, label: str) -> List[str]:
        """Normalizes input/output names to a list of strings.

        This method accepts a string, a sequence of strings, or None, and
        returns a list of strings. This is a convenience method for ensuring
        that input and output names are always in a consistent format.

        Args:
            names: The input/output names to normalize. Can be a single
                string, a sequence of strings, or None.
            label: A label for the names being normalized, used in error
                messages (e.g., "inputs", "outputs").

        Returns:
            A list of strings.

        Raises:
            TypeError: If `names` is not a string, a sequence of strings, or
                None.
        """
        if names is None:
            return []
        if isinstance(names, str):
            return [names]
        if isinstance(names, Sequence) and all(isinstance(x, str) for x in names):
            return list(names)
        raise TypeError(
            f"{label} must be a string or a sequence of strings, got {type(names).__name__}."
        )

    def validate_io(
        self,
        inputs: str | Sequence[str] | None,
        outputs: str | Sequence[str] | None,
    ) -> Tuple[List[str], List[str]]:
        """Validates and normalizes the input and output names for the operation.

        This method ensures that the number of inputs and outputs matches the
        `EXPECTED_INPUTS` and `EXPECTED_OUTPUTS` attributes of the class.

        Args:
            inputs: The names of the inputs for the operation.
            outputs: The names of the outputs for the operation.

        Returns:
            A tuple containing the normalized lists of input and output names.

        Raises:
            ValueError: If the number of inputs or outputs does not match the
                expected number.
        """
        in_list = self._normalize_names(inputs, "inputs")
        out_list = self._normalize_names(outputs, "outputs")

        if self.EXPECTED_INPUTS is not None and len(in_list) != self.EXPECTED_INPUTS:
            raise ValueError(
                f"{self.__class__.__name__}: expected {self.EXPECTED_INPUTS} input name(s), "
                f"got {len(in_list)}: {in_list!r}"
            )
        if self.EXPECTED_OUTPUTS is not None and len(out_list) != self.EXPECTED_OUTPUTS:
            raise ValueError(
                f"{self.__class__.__name__}: expected {self.EXPECTED_OUTPUTS} output name(s), "
                f"got {len(out_list)}: {out_list!r}"
            )

        return in_list, out_list

    def initialize(self):
        """Performs any subclass-specific initialization.

        This method is an optional hook for subclasses to perform any setup
        that is required before the operation is run.
        """
        return None

    @abstractmethod
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate configuration. Raise ValueError with clear message if invalid."""
        ...

    @abstractmethod
    def run(self, *sources: Any) -> Any:
        """Executes the operation on the given source(s).

        This method should be implemented by subclasses to perform the actual
        processing.

        Args:
            *sources: The input(s) for the operation.

        Returns:
            The output(s) of the operation.
        """
        ...

    def __repr__(self) -> str:
        """Returns an unambiguous, developer-oriented representation of the object."""
        return (
            f"<{self.__class__.__name__}("
            f"kind='{getattr(self, 'kind', '?')}', "
            f"type='{getattr(self, 'type_name', '?')}', "
            f"cfg={self.cfg})>"
        )

    def __str__(self) -> str:
        """Returns a human-friendly string representation of the object."""
        return f"{self.kind}:{self.type_name} {self.cfg}"
