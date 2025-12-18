import pytest
import copy
from unittest.mock import MagicMock, patch

import multiplex_pipeline.utils.config_loaders as config_loaders

@pytest.fixture
def normalization_config():
    """
    Creates a configuration dictionary mirroring the user's specific 
    image_transformer normalization scenario.
    """
    return {
        "additional_elements": [
            {
                "category": "image_transformer",
                "type": "normalize",
                # The list of channels to process
                "input": ["DAPI", "CD45", "HLA1"], 
                # The output pattern dependent on the input
                "output": "${input}_norm",
                # Nested parameters that must be preserved
                "parameters": {
                    "low": 1,
                    "high": 99.8
                },
            }
        ]
    }

def test_expand_image_transformer_normalization(normalization_config):
    """
    Verifies the expansion logic for the specific image_transformer case.
    
    Scenario:
    - Input is a list of 3 channels.
    - Output string contains ${input}.
    - Metadata (category, type, parameters) must be preserved in every expanded step.
    """
    # Execute the expansion
    expanded_cfg = config_loaders.expand_pipeline(normalization_config)
    steps = expanded_cfg["additional_elements"]

    # 1. Verify correct number of generated steps (Fan-out)
    assert len(steps) == 3

    # 2. Check the first element (DAPI)
    first_step = steps[0]
    assert first_step["input"] == "DAPI"
    # Crucial: Check if ${input} in 'output' was replaced
    assert first_step["output"] == "DAPI_norm" 
    # Check if static metadata is preserved
    assert first_step["category"] == "image_transformer"
    assert first_step["parameters"]["low"] == 1

    # 3. Check a middle element (CD45)
    second_step = steps[1]
    assert second_step["input"] == "CD45"
    assert second_step["output"] == "CD45_norm"

    # 4. Check the last element (HLA1)
    last_step = steps[2]
    assert last_step["input"] == "HLA1"
    assert last_step["output"] == "HLA1_norm"