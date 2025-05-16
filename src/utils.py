"""
utils.py - Shared utilities for BatchGrader

Includes:
- deep_merge_dicts: Recursively merge two dictionaries (for config merging)
- ensure_config_files_exist: Creates default config.yaml and prompts.yaml from examples if they don't exist.
"""
import os
import shutil
import logging
from typing import Any, Optional
from pathlib import Path

import tiktoken
from src.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def deep_merge_dicts(a, b):
    """
    Recursively merge dict b into dict a and return the result.
    Values from b take precedence over a.
    """
    if not isinstance(a, dict):
        return b
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def ensure_config_files_exist(logger):
    """
    Checks for config.yaml and prompts.yaml in the config directory.
    If they don't exist, it copies them from their respective .example files.

    Args:
        logger: An instance of the application's logger.
    """
    try:
        # Use the PROJECT_ROOT from constants to ensure consistent project structure
        config_dir = PROJECT_ROOT / "config"

        files_to_check = {
            "config.yaml": "config.yaml.example",
            "prompts.yaml": "prompts.yaml.example"
        }

        os.makedirs(str(config_dir), exist_ok=True)

        for dest_file, src_example_file in files_to_check.items():
            dest_path = config_dir / dest_file
            src_example_path = config_dir / src_example_file

            if not os.path.exists(str(dest_path)):
                if os.path.exists(str(src_example_path)):
                    shutil.copy2(src_example_path, dest_path)
                    logger.info(
                        f"'{dest_path}' not found. Copied from '{src_example_path}'."
                    )
                else:
                    logger.warning(
                        f"'{dest_path}' not found, and example file '{src_example_path}' also missing. Cannot create default configuration."
                    )
            else:
                logger.debug(f"'{dest_path}' already exists. No action taken.")

    except Exception as e:
        logger.error(f"Error ensuring config files exist: {e}", exc_info=True)


def get_encoder(
        model_name: Optional[str] = None) -> Optional[tiktoken.Encoding]:
    """
    Get a tiktoken encoder for the specified model or a default one.
    
    Args:
        model_name: Optional name of the model to get encoder for. If None, uses cl100k_base.
        
    Returns:
        tiktoken.Encoding object if successful, None if failed
        
    Note:
        This function centralizes encoder acquisition logic to avoid duplication
        across the codebase. It handles errors gracefully and logs warnings.
    """
    try:
        if model_name:
            encoder = tiktoken.encoding_for_model(model_name)
        else:
            encoder = tiktoken.get_encoding("cl100k_base")
        return encoder
    except Exception as e:
        logger.warning(f"Failed to initialize encoder: {e}")
        return None
