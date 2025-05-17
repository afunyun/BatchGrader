"""
Utility functions for BatchGrader.

This module provides various utility functions used throughout the BatchGrader application.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import tiktoken
from loguru import logger

# Import PROJECT_ROOT from constants if available
try:
    from ..constants import PROJECT_ROOT
except ImportError:
    # Fallback if constants.py is not available
    PROJECT_ROOT = Path(__file__).parent.parent.parent


def deep_merge_dicts(dict_a: Dict[Any, Any], dict_b: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merge dict_b into dict_a and return the result.
    Values from dict_b take precedence over dict_a.
    
    Args:
        dict_a: The base dictionary to merge into
        dict_b: The dictionary to merge from (takes precedence)
        
    Returns:
        A new dictionary with values from both dictionaries merged
    """
    if not isinstance(dict_a, dict):
        return dict_b
    result = dict_a.copy()
    for key, value in dict_b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def ensure_config_files_exist(log) -> None:
    """
    Checks for config.yaml and prompts.yaml in the config directory.
    If they don't exist, it copies them from their respective .example files.

    Args:
        log: An instance of the application's logger.
    """
    try:
        config_dir = PROJECT_ROOT / "config"

        files_to_check = {
            "config.yaml": "config.yaml.example",
            "prompts.yaml": "prompts.yaml.example",
        }

        os.makedirs(config_dir, exist_ok=True)

        for dest_file, src_example_file in files_to_check.items():
            dest_path = config_dir / dest_file
            src_example_path = config_dir / src_example_file

            if dest_path.exists():
                log.debug("'%s' already exists. No action taken.", dest_path)
            else:
                if src_example_path.exists():
                    shutil.copy2(src_example_path, dest_path)
                    log.info("'%s' not found. Copied from '%s'.", dest_path, src_example_path)
                else:
                    log.warning(
                        "'%s' not found, and example file '%s' also missing. "
                        "Cannot create default configuration.",
                        dest_path, src_example_path
                    )
    except (OSError, IOError) as error:
        log.error("File system error while ensuring config files exist: %s", error)


def get_encoder(model_name: Optional[str] = None) -> Optional[tiktoken.Encoding]:
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
            return tiktoken.encoding_for_model(model_name)
        return tiktoken.get_encoding("cl100k_base")
    except Exception as error:
        logger.warning("Failed to initialize encoder: %s", error)
        return None
