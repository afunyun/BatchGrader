import pytest
import pandas as pd
from batch_job import BatchJob
from unittest.mock import MagicMock, patch
from exceptions import BatchGraderError


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['First row', 'Second row', 'Third row'],
        'value': [10.1, 20.2, 30.3]
    })


@pytest.fixture
def sample_batch_job(sample_df):
    """Create a sample BatchJob instance for testing."""
    return BatchJob(chunk_id_str="test_chunk_001",
                    chunk_df=sample_df,
                    system_prompt="This is a test system prompt",
                    response_field="text",
                    original_filepath="test/input/test_file.csv",
                    chunk_file_path="test/input/_chunked/test_file_part_1.csv",
                    llm_model="gpt-4",
                    api_key_prefix="sk-test")


@pytest.fixture
def mock_job():
    """Create a mock BatchJob instance for testing."""
    return BatchJob(chunk_id_str="test_chunk_001",
                    chunk_df=pd.DataFrame(),
                    system_prompt="Test prompt",
                    response_field="text",
                    original_filepath="test.csv",
                    chunk_file_path="test_chunk.csv")


def test_batch_job_initialization(sample_df):
    """Test BatchJob initialization with various parameters."""
    # Test with required parameters only
    job = BatchJob(chunk_id_str="test_chunk_001",
                   chunk_df=sample_df,
                   system_prompt="Test prompt",
                   response_field="text",
                   original_filepath="test.csv",
                   chunk_file_path="test_chunk.csv")

    assert job.chunk_id_str == "test_chunk_001"
    assert job.system_prompt == "Test prompt"
    assert job.response_field == "text"
    assert job.status == "pending"  # Default status
    assert job.error_message is None
    assert job.result_data is None
    assert job.llm_model is None
    assert job.api_key_prefix is None

    # Test with all parameters
    job = BatchJob(chunk_id_str="test_chunk_002",
                   chunk_df=sample_df,
                   system_prompt="Test prompt",
                   response_field="text",
                   original_filepath="test.csv",
                   chunk_file_path="test_chunk.csv",
                   llm_model="gpt-4",
                   api_key_prefix="sk-test",
                   status="completed",
                   error_message="No error",
                   error_details={"code": 0},
                   result_data=sample_df.copy())

    assert job.chunk_id_str == "test_chunk_002"
    assert job.status == "completed"
    assert job.error_message == "No error"
    assert job.error_details == {"code": 0}
    assert isinstance(job.result_data, pd.DataFrame)
    assert job.llm_model == "gpt-4"
    assert job.api_key_prefix == "sk-test"

    # Test initialization with None DataFrame
    job = BatchJob(
        chunk_id_str="test_chunk_003",
        chunk_df=None,  # None DataFrame
        system_prompt="Test prompt",
        response_field="text",
        original_filepath="test.csv",
        chunk_file_path="test_chunk.csv")

    assert job.chunk_df is None


def test_batch_job_status_transitions(sample_batch_job):
    """Test BatchJob status transitions."""
    job = sample_batch_job

    assert job.status == "pending"

    # Transition to running
    job.status = "running"
    assert job.status == "running"

    # Transition to completed
    job.status = "completed"
    assert job.status == "completed"

    # Set result data
    result_df = pd.DataFrame({"result": ["Success"]})
    job.result_data = result_df
    assert job.result_data is result_df

    # Transition to error state
    job.status = "error"
    job.error_message = "Something went wrong"
    job.error_details = "Error details"

    assert job.status == "error"
    assert job.error_message == "Something went wrong"
    assert job.error_details == "Error details"


def test_batch_job_default_values():
    """Test BatchJob default values."""
    job = BatchJob(chunk_id_str="test_defaults",
                   chunk_df=pd.DataFrame(),
                   system_prompt="Test",
                   response_field="text",
                   original_filepath="original.csv",
                   chunk_file_path="chunk.csv")

    assert job.input_tokens == 0
    assert job.output_tokens == 0
    assert job.cost == 0.0
    assert job.openai_batch_id is None
    assert job.input_file_id_for_chunk is None


def test_get_status_log_str(sample_batch_job):
    """Test the get_status_log_str method."""
    job = sample_batch_job

    # Default state
    log_str = job.get_status_log_str()
    assert "Chunk test_chunk_001" in log_str
    assert "pending" in log_str

    # With batch ID
    job.openai_batch_id = "batch_123456"
    log_str = job.get_status_log_str()
    assert "Batch ID: batch_123456" in log_str

    # With error
    job.status = "error"
    job.error_message = "API connection failed"
    log_str = job.get_status_log_str()
    assert "error" in log_str
    assert "API connection failed" in log_str


def test_batch_job_token_tracking(sample_batch_job):
    """Test token tracking attributes in BatchJob."""
    job = sample_batch_job

    # Initial values
    assert job.input_tokens == 0
    assert job.output_tokens == 0

    # Update token counts
    job.input_tokens = 100
    job.output_tokens = 50
    job.cost = 0.25

    assert job.input_tokens == 100
    assert job.output_tokens == 50
    assert job.cost == 0.25


def test_batch_job_with_invalid_file() -> None:
    """Test BatchJob initialization with invalid parameters.\n\n    Args:\n        None\n\n    Returns:\n        None\n    """
    # Create a job with nonexistent file, but verify it works
    # Note: Based on the implementation, BatchJob doesn't validate file existence
    job = BatchJob(chunk_id_str="invalid",
                   chunk_df=None,
                   system_prompt="test",
                   response_field="text",
                   original_filepath="nonexistent.csv",
                   chunk_file_path="nonexistent_chunk.csv")

    assert job.chunk_id_str == "invalid"
    assert job.chunk_df is None
    assert job.status == "pending"
