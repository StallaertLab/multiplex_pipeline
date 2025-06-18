import requests
from io import BytesIO
import globus_sdk
import json
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from typing import Dict


@dataclass
class GlobusConfig:
    """Configuration for Globus client and endpoints."""
    client_id: str
    r_collection_id: str
    local_collection_id: str
    transfer_tokens: Dict
    https_token: Dict
    https_server: Dict

    @classmethod
    def from_config_files(cls, config_dir: Path) -> 'GlobusConfig':
        """Load Globus configuration from JSON files in the specified directory."""
        try:
            with open(config_dir / "globus_config.json", "r") as f:
                config = json.load(f)
                client_id = config["client_id"]
                r_collection_id = config["r_collection_id"]
                local_collection_id = config["local_collection_id"]
            
            with open(config_dir / "globus_tokens.json", "r") as f:
                transfer_tokens = json.load(f)
            
            with open(config_dir / "globus_https_tokens.json", "r") as f:
                https_token = json.load(f)
            
            with open(config_dir / "globus_https_server.json", "r") as f:
                https_server = json.load(f)
            
            return cls(
                client_id=client_id,
                r_collection_id=r_collection_id,
                local_collection_id=local_collection_id,
                transfer_tokens=transfer_tokens,
                https_token=https_token,
                https_server=https_server
            )
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise


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
        transfer_rt, auth_client, access_token=transfer_at, expires_at=expires_at_s
    )

    # create TransferClient 
    tc = globus_sdk.TransferClient(authorizer=authorizer)

    return tc

def get_with_globus_https(file_path, https_server, https_token):
    """
    Get a file from a Globus endpoint using a Globus access token.
    input:
        file_path: str
        https_server: str
        https_token: str # Globus access token
    output:
        BytesIO object
    """

    headers = {
    'Authorization': f'Bearer {https_token}',
    'Accept': 'application/octet-stream'
    }

    https_url = f"{https_server}{file_path}"

    response = requests.get(https_url, headers=headers, allow_redirects=True)
    response.raise_for_status()  # Ensure the request was successful

    return BytesIO(response.content)