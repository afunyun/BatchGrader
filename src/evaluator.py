"""
name is misleading because hindsight is 20/20, but this basically just loads the prompts and prepares the prompts for the main batch request constructor so it can add them to the .jsonl
"""
import sys

import yaml


def load_prompt_template(name='evaluation_prompt'):
    """
    Loads the prompt template from prompts.yaml. If the prompt is missing, falls back to DEFAULT_PROMPTS in config_loader.py and sends a message in console to fix the config.
    """
    import sys
    try:
        with open("config/prompts.yaml", 'r') as f:
            data = yaml.safe_load(f) or {}
        if name in data:
            return data[name]
        else:
            print(
                f"[WARN] Prompt '{name}' not found in prompts.yaml. Using default.",
                file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not load prompts.yaml ({e}). Using default.",
              file=sys.stderr)
    # fallback to default
    try:
        from config_loader import DEFAULT_PROMPTS
        if name in DEFAULT_PROMPTS:
            return DEFAULT_PROMPTS[name]
        else:
            raise KeyError(
                f"Prompt '{name}' not found in DEFAULT_PROMPTS. This should never happen."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load prompt '{name}' from both prompts.yaml and DEFAULT_PROMPTS: {e}, this should REALLY never happen. It's joever."
        )


# def prepare_prompt(template, response):
#     return template.format(response=response)

# def prepare_batch_prompt(template, responses):
#     batch_list = '\n'.join(f"{i+1}: {resp}" for i, resp in enumerate(responses))
#     return template.format(batch_list=batch_list)
