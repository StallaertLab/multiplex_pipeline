from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal, Type, Annotated, TYPE_CHECKING

from pydantic import BaseModel, Field, create_model, model_validator, ValidationInfo
from multiplex_pipeline.processors.registry import REGISTRY, Kind

if TYPE_CHECKING:
    from spatialdata import SpatialData

from loguru import logger

###################################################################
# Top-Level Configuration Models
###################################################################

class GeneralSettings(BaseModel):
    image_dir: str
    analysis_name: str
    local_analysis_dir: str
    remote_analysis_dir: str
    log_dir: Optional[Path] = None

class CoreDetectionSettings(BaseModel):
    detection_image: str
    core_info_file_path: Optional[str] = None
    im_level: int
    min_area: int
    max_area: int
    min_iou: float
    min_st: float
    min_int: int
    frame: int

class CoreCuttingSettings(BaseModel):
    cores_dir_tif: Optional[str] = None
    cores_dir_output: Optional[str] = None
    include_channels: Optional[Union[str, List[str]]] = None
    exclude_channels: Optional[Union[str, List[str]]] = None
    use_markers: Optional[Union[str, List[str]]] = None
    ignore_markers: Optional[Union[str, List[str]]] = None
    margin: int
    mask_value: int
    transfer_cleanup_enabled: bool
    core_cleanup_enabled: bool

class QuantTask(BaseModel):
    name: str
    masks: Dict[str, str]
    layer_connection: str = None

class StorageSettings(BaseModel):
    chunk_size: List[int]
    max_pyramid_level: int
    downscale: int

###################################################################
# Pipeline Step Models (for 'additional_elements')
###################################################################
class BaseStep(BaseModel):
    category: Kind
    type: str
    input: Union[str, List[str]]
    output: Union[str, List[str]]
    keep: bool = True

def create_step_models() -> list[Type[BaseModel]]:
    """Dynamically creates Pydantic models for each registered processor."""
    all_step_models = []
    for kind, processors in REGISTRY.items():
        for name, entry in processors.items():
            # 3. Create a unique model for each specific step, e.g., "RingStep"
            model_name = f"{name.capitalize()}Step"
            
            step_model = create_model(
                model_name,
                # It must have this specific category and type
                category=(Literal[kind], ...),
                type=(Literal[name], ...),
                # Its parameters must match the processor's Params model
                parameters=(entry.param_model, Field(default={})),
                # It inherits the common fields from BaseStep
                __base__=BaseStep,
            )
            all_step_models.append(step_model)
    return all_step_models

# 4. PipelineStep is a Union of all dynamically generated models
step_models = create_step_models()
if not step_models:
    PipelineStep = BaseStep # Fallback for an empty registry
else:
    PipelineStep = Union[tuple(step_models)]

class AnalysisConfig(BaseModel):
    """The root model for the entire Track Gardener configuration.

    This class acts as the main entry point for parsing and validating the
    complete YAML configuration file.
    """

    general: GeneralSettings
    core_detection: CoreDetectionSettings
    core_cutting: CoreCuttingSettings
    additional_elements: List[Annotated[PipelineStep, Field(discriminator="type")]]
    quant: List[QuantTask]
    sdata_storage: StorageSettings

    analysis_dir: Path = Path(".")
    temp_dir: Path = Path(".")
    log_dir_path: Path = Path(".")
    core_info_file_path: Path = Path(".")
    cores_dir_tif_path: Path = Path(".")
    cores_dir_output_path: Path = Path(".")

    @model_validator(mode='after')
    def _resolve_paths(self, info: ValidationInfo) -> AnalysisConfig:
        """
        This validator resolves paths and sets defaults.
        """
        remote = info.context.get("remote_analysis", False)

        # Determine base analysis directory
        base_dir_str = self.general.remote_analysis_dir if remote else self.general.local_analysis_dir
        analysis_dir = Path(base_dir_str) / self.general.analysis_name
        self.analysis_dir = analysis_dir

        # Define defaults
        defaults = {
            "log_dir": analysis_dir / "logs",
            "core_info_file_path": analysis_dir / "cores.csv",
            "cores_dir_tif": analysis_dir / "temp",
            "cores_dir_output": analysis_dir / "cores",
            "temp_dir": analysis_dir / "temp",
        }

        # Populate final Path objects
        self.log_dir_path = Path(self.general.log_dir or defaults["log_dir"])
        
        self.core_info_file_path = Path(self.core_detection.core_info_file_path 
        or defaults["core_info_file_path"])
        
        self.cores_dir_tif_path = Path(self.core_cutting.cores_dir_tif or defaults["cores_dir_tif"])
        
        self.cores_dir_output_path = Path(self.core_cutting.cores_dir_output or defaults["cores_dir_output"])
        
        self.temp_dir = defaults["temp_dir"]

        return self
    
    def validate_pipeline(self, sdata: "SpatialData") -> None:
        """
        Validates the pipeline's data flow against a SpatialData object.

        This method checks that for every step:
        1. All required inputs exist.
        2. Inputs can be from the initial SpatialData object or outputs of prior steps.

        Raises:
            ValueError: If an input is not found at any stage.
        """
        # Start with the set of layers available in the initial sdata object.
        available_layers = set(sdata.images) | set(sdata.labels)

        for i, step in enumerate(self.additional_elements):
            # Normalize step inputs to a list for consistent processing
            inputs = [step.input] if isinstance(step.input, str) else step.input
            
            # Check if all inputs for the current step are available
            for required_input in inputs:
                if required_input not in available_layers:
                    msg = f"Pipeline validation failed at step {i} ('{step.type}'):\n \
                        Input '{required_input}' not found.\n \
                        Available layers: {sorted(list(available_layers))}"
                    logger.error(msg)
                    raise ValueError(msg)

            # Add the outputs of the current step to the set of available layers
            outputs = [step.output] if isinstance(step.output, str) else step.output
            available_layers.update(outputs)

        # If the loop completes without errors, the pipeline is valid.
        logger.info("✅ Pipeline validation successful.")

