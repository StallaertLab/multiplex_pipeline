import yaml
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict

import globus_sdk
import requests
from loguru import logger


@dataclass
class GlobusConfig:
    """Configuration for Globus client and endpoints."""

    client_id: str
    source_collection_id: str
    destination_collection_id: str
    transfer_tokens: Dict

    @classmethod
    def from_config_files(cls, config_path: str | Path, from_collection: str, to_collection: str) -> "GlobusConfig":
        """Load Globus configuration from JSON files in the specified directory."""
        
        # check that config exists
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"Configuration file not found.")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        

        with open(config_path) as f:
            config = yaml.safe_load(f)
            client_id = config["gc"]["client_id"]
            source_collection_id = config["gc"][from_collection]
            destination_collection_id = config["gc"][to_collection]
            transfer_tokens = config['transfer_tokens']

        return cls(
            client_id=client_id,
            source_collection_id=source_collection_id,
            destination_collection_id=destination_collection_id,
            transfer_tokens=transfer_tokens
        )

def create_globus_tc(client_id, transfer_tokens):
    """
    Create a TransferClient object using the Globus SDK.
    """

    auth_client = globus_sdk.NativeAppAuthClient(client_id)

    transfer_rt = transfer_tokens["refresh_token"]
    transfer_at = transfer_tokens["access_token"]
    expires_at_s = transfer_tokens["expires_at_seconds"]

    # construct a RefreshTokenAuthorizer
    # note that `client` is passed to it, to allow it to do the refreshes
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        transfer_rt,
        auth_client,
        access_token=transfer_at,
        expires_at=expires_at_s,
    )

    # create TransferClient
    tc = globus_sdk.TransferClient(authorizer=authorizer)

    return tc
