"""
utils.py - Shared utilities for BatchGrader

Includes:
- deep_merge_dicts: Recursively merge two dictionaries (for config merging)
"""


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
