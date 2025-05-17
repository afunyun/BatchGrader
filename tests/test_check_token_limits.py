import pandas as pd
import pytest

from batchgrader.file_processor import check_token_limits


class MockEncoder:
    """A mock encoder that treats each character as a token."""
    def encode(self, text: str) -> list:
        return list(text) # len(self.encode(text)) will be len(text)

# The 'encoder' parameter for dummy_encoder_func_generator was unused.
# It's removed to avoid confusion, as the actual 'encoder' passed to
# check_token_limits will now be an instance of MockEncoder.
def dummy_encoder_func_generator(prompt_content_for_len_calc, response_field_name_for_lookup):
    """
    Generates a token counting function based on string lengths.
    This was the original logic for the dummy encoder, but check_token_limits
    now expects an encoder object with an .encode() method.
    This function is NO LONGER USED by the tests below directly with check_token_limits,
    but kept here for reference if other tests relied on its specific counting.
    The MockEncoder above is now used.
    """
    def counter(row):
        # This logic was: len(system_prompt) + len(response_in_row)
        return len(prompt_content_for_len_calc) + len(str(row[response_field_name_for_lookup]))
    return counter


# Test cases for error handling in check_token_limits function.
ERROR_CASES = [
    pytest.param("not_df",
                 "prompt",
                 "resp",
                 MockEncoder(), # Use instance of MockEncoder
                 10,
                 TypeError, # This test case might change if None is passed to create_token_counter
                 "df must be a pandas DataFrame",
                 id="not-a-dataframe"),
    pytest.param(pd.DataFrame(),
                 "prompt",
                 "resp",
                 MockEncoder(),
                 10,
                 ValueError,
                 "DataFrame cannot be empty",
                 id="empty-dataframe"),
    pytest.param(pd.DataFrame({"resp": [1]}),
                 "",
                 "resp",
                 MockEncoder(),
                 10,
                 ValueError,
                 "system_prompt_content must be a non-empty string",
                 id="empty-system-prompt"),
    pytest.param(pd.DataFrame({"resp": [1]}),
                 "prompt",
                 "",
                 MockEncoder(),
                 10,
                 ValueError,
                 "response_field must be a non-empty string",
                 id="empty-response-field"),
    pytest.param(pd.DataFrame({"a": [1]}),
                 "prompt",
                 "resp",
                 MockEncoder(),
                 10,
                 ValueError,
                 "response_field 'resp' not found",
                 id="response-field-not-found"),
    pytest.param(pd.DataFrame({"resp": [1]}),
                 "prompt",
                 "resp",
                 MockEncoder(),
                 0,
                 ValueError,
                 "token_limit must be a positive integer",
                 id="zero-token-limit"),
    pytest.param(pd.DataFrame({"resp": [1]}),
                 "prompt",
                 "resp",
                 None,
                 10,
                 ValueError,
                 "encoder cannot be None",
                 id="encoder-is-none"),
]


@pytest.mark.parametrize(
    "df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr",
    ERROR_CASES,
)
def test_check_token_limits_errors(df, system_prompt_content, response_field,
                                   encoder, token_limit, exc, msg_substr):
    """
    Test check_token_limits for invalid or edge-case inputs.

    Verifies that appropriate exceptions are raised with informative messages
    for various misconfigurations and invalid arguments.
    """
    with pytest.raises(exc) as exc_info:
        check_token_limits(
            df,
            system_prompt_content,
            response_field,
            encoder,
            token_limit,
            raise_on_error=True,
        )
    # Use explicit assertion message for clarity in test failures
    assert msg_substr in str(
        exc_info.value), (f"Expected error message containing '{msg_substr}', "
                          f"but got: {str(exc_info.value)}")


def test_check_token_limits_success_under_limit():
    df = pd.DataFrame({"resp": ["a", "bb", "ccc"]})
    encoder_instance = MockEncoder()
    is_valid, stats = check_token_limits(df,
                                         "X", # system_prompt_content
                                         "resp", # response_field
                                         encoder_instance, # actual encoder object
                                         100, # token_limit
                                         raise_on_error=True)
    assert is_valid
    assert isinstance(stats, dict)
    # Expected: (len("X") + len("\n") + len("a")) + (len("X") + len("\n") + len("bb")) + (len("X") + len("\n") + len("ccc"))
    # (1 + 1 + 1) + (1 + 1 + 2) + (1 + 1 + 3) = 3 + 4 + 5 = 12
    assert stats["total"] == 12.0


def test_check_token_limits_exceed_limit():
    df = pd.DataFrame({"resp": ["aaaa"]})
    encoder_instance = MockEncoder()
    # token count for "Z\naaaa" using MockEncoder = len("Z") + len("\n") + len("aaaa") = 1 + 1 + 4 = 6
    # limit = 4. Since 6 > 4, is_valid should be False.
    is_valid, stats = check_token_limits(df,
                                         "Z", # system_prompt_content
                                         "resp", # response_field
                                         encoder_instance, # actual encoder object
                                         4,    # token_limit
                                         raise_on_error=False)
    assert not is_valid
    assert stats["total"] == 6.0
