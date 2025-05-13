"""
utils.py - Shared utilities for BatchGrader

Includes:
- deep_merge_dicts: Recursively merge two dictionaries (for config merging)
- ensure_config_files_exist: Creates default config.yaml and prompts.yaml from examples if they don't exist.
"""
import os
import shutil


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
        # Determine the project root directory (assuming utils.py is in src/)
        current_script_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_script_path)
        project_root = os.path.dirname(src_dir)
        config_dir = os.path.join(project_root, "config")

        files_to_check = {
            "config.yaml": "config.yaml.example",
            "prompts.yaml": "prompts.yaml.example"
        }

        os.makedirs(config_dir, exist_ok=True)

        for dest_file, src_example_file in files_to_check.items():
            dest_path = os.path.join(config_dir, dest_file)
            src_example_path = os.path.join(config_dir, src_example_file)

            if not os.path.exists(dest_path):
                if os.path.exists(src_example_path):
                    shutil.copy2(src_example_path, dest_path)
                    logger.info(f"'{dest_path}' not found. Copied from '{src_example_path}'.")
                else:
                    logger.warning(f"'{dest_path}' not found, and example file '{src_example_path}' also missing. Cannot create default configuration.")
            else:
                logger.debug(f"'{dest_path}' already exists. No action taken.")

    except Exception as e:
        logger.error(f"Error ensuring config files exist: {e}", exc_info=True)
        # Depending on desired behavior, might re-raise or exit
        # For now, just log and continue, assuming config loader will fail later if files are critical
