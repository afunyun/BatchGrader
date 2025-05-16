"""
Evaluator module for loading and preparing prompt templates.
This module handles prompt loading from configuration files or falls back to defaults.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Import default prompts from config_loader instead of defining them here
from config_loader import DEFAULT_PROMPTS


def load_prompt_template(name: str = 'evaluation_prompt',
                         config_dir: Optional[Path] = None) -> str:
    """
    Loads the prompt template from prompts.yaml. If the prompt is missing, falls back to 
    DEFAULT_PROMPTS in config_loader.py and logs a warning message.
    
    Args:
        name: Name of the prompt to load
        config_dir: Optional directory path where prompts.yaml is located
        
    Returns:
        The prompt template as a string
        
    Raises:
        RuntimeError: If the prompt can't be loaded from either source
    """
    if config_dir is None:
        config_dir = Path("config")

    prompts_path = config_dir / "prompts.yaml"

    try:
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            if name in data:
                return data[name]
            else:
                print(
                    f"[WARN] Prompt '{name}' not found in {prompts_path}. Using default.",
                    file=sys.stderr)
        else:
            print(
                f"[WARN] Prompts file not found at {prompts_path}. Using default.",
                file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not load {prompts_path} ({e}). Using default.",
              file=sys.stderr)

    # Fallback to default prompts from config_loader
    if name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[name]
    else:
        raise RuntimeError(
            f"Failed to load prompt '{name}'. Not found in {prompts_path} or DEFAULT_PROMPTS. This should REALLY never happen."
        )


# def prepare_prompt(template, response):
#     return template.format(response=response)

# def prepare_batch_prompt(template, responses):
#     batch_list = '\n'.join(f"{i+1}: {resp}" for i, resp in enumerate(responses))
#     return template.format(batch_list=batch_list)
