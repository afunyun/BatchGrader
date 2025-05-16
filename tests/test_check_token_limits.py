import pytest
import pandas as pd
from src.file_processor import check_token_limits

def dummy_encoder(prompt, resp, encoder):
    # simple count: sum of prompt and response lengths
    def counter(row):
        return len(prompt) + len(str(row[resp]))
    return counter

@ pytest.mark.parametrize("df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr", [
    ("not_df", "prompt", "resp", dummy_encoder, 10, TypeError, "df must be a pandas DataFrame"),
    (pd.DataFrame(), "prompt", "resp", dummy_encoder, 10, ValueError, "DataFrame cannot be empty"),
    (pd.DataFrame({'resp':[1]}), "", "resp", dummy_encoder, 10, ValueError, "system_prompt_content must be a non-empty string"),
    (pd.DataFrame({'resp':[1]}), "prompt", "", dummy_encoder, 10, ValueError, "response_field must be a non-empty string"),
    (pd.DataFrame({'a':[1]}), "prompt", "resp", dummy_encoder, 10, ValueError, "response_field 'resp' not found"),
    (pd.DataFrame({'resp':[1]}), "prompt", "resp", dummy_encoder, 0, ValueError, "token_limit must be a positive integer"),
    (pd.DataFrame({'resp':[1]}), "prompt", "resp", None, 10, ValueError, "encoder cannot be None"),
])
def test_check_token_limits_errors(df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr):
    with pytest.raises(exc) as e:
        check_token_limits(df, system_prompt_content, response_field, encoder, token_limit, raise_on_error=True)
    assert msg_substr in str(e.value)


def test_check_token_limits_success_under_limit():
    df = pd.DataFrame({'resp':["a","bb","ccc"]})
    is_valid, stats = check_token_limits(df, "X", "resp", dummy_encoder, 100, raise_on_error=True)
    assert is_valid
    assert isinstance(stats, dict)
    assert stats['total'] == float((len("X")+1)+(len("X")+2)+(len("X")+3))


def test_check_token_limits_exceed_limit():
    df = pd.DataFrame({'resp':["aaaa"]})
    # token count = len(prompt)+len(resp)=1+4=5, limit=4 => exceed
    is_valid, stats = check_token_limits(df, "Z", "resp", dummy_encoder, 4, raise_on_error=False)
    assert not is_valid
    assert stats['total'] == float(len("Z")+4)
