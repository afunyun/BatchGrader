"""
Token counting and limit checking utilities.

This module provides functions for counting tokens in text content, checking
token limits, and calculating token statistics.
"""

import logging
from typing import Any, Callable, Dict, Optional

from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class TokenLimitExceededError(Exception):
    """Raised when token limit is exceeded."""

    pass


def count_tokens_in_content(content: str, encoder: Any) -> int:
    """
    Count the number of tokens in a given content string using the
    provided encoder.

    Args:
        content: The text content to count tokens for
        encoder: The tokenizer/encoder to use for tokenization

    Returns:
        int: Number of tokens in the content
    """
    try:
        return len(encoder.encode(str(content)))
    except (AttributeError, TypeError, Exception) as e:
        logger.error(f"Error encoding content: {e}")
        return 0


def create_token_counter(
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    prompt_template: Optional[str] = None,
) -> Callable[[Dict[str, Any]], int]:
    """
    Create a token counter function for processing DataFrame rows.

    Args:
        system_prompt_content: Content of the system prompt
        response_field: Field name containing user content
        encoder: The tokenizer/encoder to use
        prompt_template: Optional template string for custom prompt formatting.
            Use '{system}' for system prompt and '{user}' for user content.

    Returns:
        Callable: A function that takes a row dict and returns token count
    """

    def token_counter(row: Dict[str, Any]) -> int:
        user_content = str(row.get(response_field, ""))
        system_content = system_prompt_content

        if prompt_template:
            prompt_text = prompt_template.format(system=system_content,
                                                 user=user_content)
        else:
            prompt_text = f"{system_content}\n{user_content}"

        return count_tokens_in_content(prompt_text, encoder)

    return token_counter


def count_tokens_in_df(
    data_frame: DataFrame,
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    **kwargs: Any,
) -> Series:
    """
    Count tokens for each row in a DataFrame.

    Args:
        data_frame: Input DataFrame
        system_prompt_content: Field name or content for system prompt
        response_field: Field name containing user content
        encoder: The tokenizer/encoder to use
        **kwargs: Additional arguments passed to create_token_counter

    Returns:
        Series: Token counts for each row
    """
    token_counter = create_token_counter(
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        encoder=encoder,
        **kwargs,
    )
    return data_frame.apply(token_counter, axis=1)


def calculate_token_stats(token_counts: Series) -> Dict[str, float]:
    """
    Calculate token statistics from a Series of token counts.

    Args:
        token_counts: Series containing token counts per row

    Returns:
        Dict containing total, average, and max token counts
    """
    return {
        "total": float(token_counts.sum()),
        "average": float(token_counts.mean()),
        "max": float(token_counts.max()),
    }


def check_token_limit(token_stats: Dict[str, float],
                      token_limit: int,
                      raise_on_exceed: bool = False) -> bool:
    """
    Check if token usage exceeds the specified limit.

    Args:
        token_stats: Dictionary containing token statistics
        token_limit: Maximum allowed tokens
        raise_on_exceed: If True, raises TokenLimitExceededError when limit
                         is exceeded

    Returns:
        bool: True if under limit, False if over limit

    Raises:
        TokenLimitExceededError: If raise_on_exceed is True and limit is
                                 exceeded
    """
    is_under_limit = token_stats["total"] <= token_limit

    if not is_under_limit and raise_on_exceed:
        raise TokenLimitExceededError(
            f"Token limit exceeded: {token_stats['total']} > {token_limit}")

    return is_under_limit


def get_token_count_message(token_stats: Dict[str, float]) -> str:
    """
    Generate a formatted message with token statistics.

    Args:
        token_stats: Dictionary containing token statistics

    Returns:
        Formatted string with token statistics
    """
    return (f"[TOKEN COUNT] "
            f"Total: {int(token_stats['total'])}, "
            f"Avg: {token_stats['average']:.1f}, "
            f"Max: {int(token_stats['max'])}")


def count_completion_tokens(row, encoder, field="llm_score"):
    """Count tokens in the LLM completion field (default: 'llm_score')."""
    completion = row.get(field, "")
    if completion is None:
        return 0
    return len(encoder.encode(str(completion)))


def count_input_tokens(row, system_prompt, field, encoder):
    """Count tokens for input including system prompt and user content."""
    sys_tokens = len(encoder.encode(system_prompt))
    user_prompt = f"Please evaluate the following text: {str(row.get(field, ''))}"
    user_tokens = len(encoder.encode(user_prompt))
    return sys_tokens + user_tokens
