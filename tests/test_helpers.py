import os


def dummy_token_counter(row):
    try:
        # If row is a pandas Series with 'text' field
        return len(row["text"])
    except (TypeError, KeyError):
        # Fallback to string length
        return len(str(row))


TEST_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "output"))
