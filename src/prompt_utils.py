import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

from config_loader import CONFIG_DIR, is_examples_file_default
from evaluator import load_prompt_template
from constants import DEFAULT_PROMPTS_FILE

logger = logging.getLogger(__name__)


def load_prompts(prompts_file: Path = DEFAULT_PROMPTS_FILE) -> Dict[str, str]:
    """
    Load prompts from a YAML file.

    Args:
        prompts_file: Path to the prompts YAML file.
                      Defaults to DEFAULT_PROMPTS_FILE.

    Returns:
        A dictionary of prompt names to prompt texts.

    Raises:
        FileNotFoundError: If the prompts file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not prompts_file.exists():
        logger.error(f"Prompts file not found: {prompts_file}")
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        if not isinstance(prompts, dict):
            logger.error(
                f"Invalid format in prompts file: {prompts_file}. Expected a dictionary."
            )
            # Consider raising a more specific error or returning empty dict based on desired behavior
            raise yaml.YAMLError(
                f"Invalid format in prompts file: {prompts_file}. Expected a dictionary."
            )
        logger.info(f"Prompts loaded successfully from {prompts_file}")
        return prompts
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {prompts_file}: {e}")
        raise  # Re-raise the YAMLError
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading prompts from {prompts_file}: {e}"
        )
        raise  # Re-raise other unexpected errors


def load_system_prompt(config: dict) -> str:
    """
    Load and format the system prompt based on the examples configuration.
    If the examples file is the default, use the generic prompt template.
    Otherwise, format dynamic examples into the prompt template.

    Args:
        config (dict): Configuration dict containing 'examples_dir'.

    Returns:
        str: The formatted system prompt.

    Raises:
        ValueError: If 'examples_dir' is missing or template placeholder is missing.
        FileNotFoundError: If examples file not found.
    """
    examples_dir = config.get('examples_dir')
    if not examples_dir:
        raise ValueError("'examples_dir' not found in config.")

    project_root = CONFIG_DIR.parent
    abs_examples_path = (project_root / examples_dir).resolve()

    # Use generic prompt if examples file is default
    if is_examples_file_default(abs_examples_path):
        return load_prompt_template("batch_evaluation_prompt_generic")

    # Load and format custom examples
    if not abs_examples_path.exists():
        raise FileNotFoundError(
            f"Examples file not found: {abs_examples_path}.")

    with open(abs_examples_path, 'r', encoding='utf-8') as ex_f:
        example_lines = [line.strip() for line in ex_f if line.strip()]
    if not example_lines:
        raise ValueError(f"No examples found in {abs_examples_path}.")

    formatted_examples = '\n'.join(f"- {ex}" for ex in example_lines)
    system_prompt_template = load_prompt_template("batch_evaluation_prompt")
    if '{dynamic_examples}' not in system_prompt_template:
        raise ValueError(
            "'{dynamic_examples}' placeholder missing in prompt template.")

    return system_prompt_template.format(dynamic_examples=formatted_examples)
