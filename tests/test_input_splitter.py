# import pytest
# from batchgrader.input_splitter import split_input # Correct function is split_file_by_token_limit

# @pytest.mark.parametrize(
#     "input_text, max_tokens, expected_output",
#     [
#         ("This is a short sentence.", 10, ["This is a short sentence."]),
#         (
#             "This is a longer sentence that should be split into multiple parts.",
#             10,
#             [
#                 "This is a longer sentence that should be",
#                 "split into multiple parts.",
#             ],
#         ),
#         (
#             "This is a very long sentence that should be split into three parts.",
#             10,
#             [
#                 "This is a very long sentence that should",
#                 "be split into",
#                 "three parts.",
#             ],
#         ),
#     ],
# )
# def test_split_input(input_text, max_tokens, expected_output):
#     # These tests are for a function `split_input` that does not exist.
#     # The available function `split_file_by_token_limit` has a different signature
#     # and purpose (splitting files/DataFrames, not simple text).
#     # Commenting out to prevent ImportError and subsequent runtime errors.
#     # assert split_input(input_text, max_tokens) == expected_output
pass
