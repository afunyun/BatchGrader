from pathlib import Path

from config_loader import CONFIG_DIR, is_examples_file_default
from evaluator import load_prompt_template
from logger import logger


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
