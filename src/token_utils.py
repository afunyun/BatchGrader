import logging
from typing import Callable, Any, Dict, Union, Optional, Tuple
from pandas import DataFrame, Series
from logger import logger

class TokenLimitExceededError(Exception):
    """Raised when token limit is exceeded"""
    pass


def count_tokens_in_content(content: str, encoder: Any) -> int:
    """
    Count the number of tokens in a given content string using the provided encoder.

    Args:
        content: The text content to count tokens for
        encoder: The tokenizer/encoder to use for tokenization
        
    Returns:
        int: Number of tokens in the content, or 0 if encoding fails
    """
    try:
        return len(encoder.encode(str(content)))
    except Exception as e:
        logger.error(f"Token encoding failed for content: {str(content)[:50]}... : {e}")
        return 0


def count_input_tokens(row: Dict[str, Any], system_prompt: str, response_field: str, encoder: Any) -> int:
    """
    Count tokens for a user prompt input row, including system prompt and user content.

    Args:
        row: Dictionary containing the data row with user content
        system_prompt: The system prompt text to prepend
        response_field: Field name in the row containing user content
        encoder: The tokenizer/encoder to use
        
    Returns:
        int: Total token count including both system and user content
    """
    sys_tokens = count_tokens_in_content(system_prompt, encoder)
    user_prompt = f"Please evaluate the following text: {str(row.get(response_field, ''))}"
    user_tokens = count_tokens_in_content(user_prompt, encoder)
    return sys_tokens + user_tokens


def count_completion_tokens(row: Dict[str, Any], encoder: Any, field: str = 'llm_score') -> int:
    """
    Count tokens in the LLM completion stored in the specified field.

    Args:
        row: Dictionary containing the data row with completion content
        encoder: The tokenizer/encoder to use
        field: Field name containing the completion text (default: 'llm_score')
        
    Returns:
        int: Number of tokens in the completion
    """
    completion = row.get(field, '')
    return count_tokens_in_content(completion, encoder)


def create_token_counter(
    system_prompt_content: str, 
    response_field: str, 
    encoder: Any,
    prompt_template: Optional[str] = None
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
        user_content = str(row.get(response_field, ''))
        system_content = system_prompt_content 
        
        if prompt_template:
            prompt_text = prompt_template.format(
                system=system_content,
                user=user_content
            )
        else:
            prompt_text = f"{system_content}\n{user_content}"
            
        return len(encoder.encode(prompt_text))
        
    return token_counter


def count_tokens_in_df(
    df: DataFrame, 
    system_prompt_content: str, 
    response_field: str, 
    encoder: Any,
    **kwargs: Any
) -> Series:
    """
    Count tokens for each row in a DataFrame.

    Args:
        df: Input DataFrame
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
        **kwargs
    )
    return df.apply(token_counter, axis=1)


def calculate_token_stats(token_counts: Series) -> Dict[str, float]:
    """
    Calculate token statistics from a Series of token counts.
    
    Args:
        token_counts: Series containing token counts per row
        
    Returns:
        Dict containing total, average, and max token counts
    """
    return {
        'total': float(token_counts.sum()),
        'average': float(token_counts.mean()),
        'max': float(token_counts.max())
    }


def check_token_limit(
    token_stats: Dict[str, float], 
    token_limit: int, 
    raise_on_exceed: bool = False
) -> bool:
    """
    Check if token usage exceeds the specified limit.
    
    Args:
        token_stats: Dictionary containing token statistics from calculate_token_stats
        token_limit: Maximum allowed tokens
        raise_on_exceed: If True, raises TokenLimitExceededError when limit is exceeded
        
    Returns:
        bool: True if under limit, False if over limit
        
    Raises:
        TokenLimitExceededError: If raise_on_exceed is True and limit is exceeded
    """
    is_under_limit = token_stats['total'] <= token_limit
    
    if not is_under_limit and raise_on_exceed:
        raise TokenLimitExceededError(
            f"Token limit exceeded: {token_stats['total']} > {token_limit}"
        )
        
    return is_under_limit


def get_token_count_message(token_stats: Dict[str, float]) -> str:
    """
    Generate a formatted message with token statistics.
    
    Args:
        token_stats: Dictionary containing token statistics
        
    Returns:
        Formatted string with token statistics
    """
    return (
        f"[TOKEN COUNT] "
        f"Total: {int(token_stats['total'])}, "
        f"Avg: {token_stats['average']:.1f}, "
        f"Max: {int(token_stats['max'])}"
    )