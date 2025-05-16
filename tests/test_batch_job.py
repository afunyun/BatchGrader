import pytest
import pandas as pd
from src.batch_job import BatchJob
from unittest.mock import MagicMock, patch
from src.exceptions import BatchGraderError


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


@pytest.fixture
def zero_item_batch_job():
    """Create a BatchJob instance with a zero-item DataFrame for testing."""
    return BatchJob(
        chunk_id_str="test_zero_item_chunk",
        chunk_df=pd.DataFrame(),  # Empty DataFrame
        system_prompt="Test system prompt for zero items",
        response_field="output_text",
        original_filepath="test/input/zero_item_file.csv",
        chunk_file_path="test/input/_chunked/zero_item_file_part_1.csv"
    )


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
    """Test BatchJob initialization with invalid parameters."""
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


def test_update_progress(sample_batch_job, zero_item_batch_job):
    """Test progress tracking and ETA calculation."""
    job = sample_batch_job
    assert job.processed_items == 0
    assert job.start_time is None
    assert job.estimated_completion_time is None

    # First update should set start time
    job.update_progress(1)
    assert job.processed_items == 1
    assert job.start_time is not None
    assert job.estimated_completion_time is not None

    # Test capping at total_items
    job.update_progress(10)
    assert job.processed_items == job.total_items

    # Test with zero items using fixture
    zero_item_batch_job.update_progress(1)
    assert zero_item_batch_job.processed_items == 0
    assert zero_item_batch_job.estimated_completion_time is None


def test_get_progress_eta_str(sample_batch_job, zero_item_batch_job):
    """Test progress string formatting in different states."""
    job = sample_batch_job

    # Initial state
    assert "0.00%" in job.get_progress_eta_str()
    assert "Calculating" in job.get_progress_eta_str()

    # After starting
    job.status = "running"
    eta_str = job.get_progress_eta_str()
    assert "0.00%" in eta_str
    assert "Started, calculating ETA" in eta_str

    # After some progress
    job.update_progress(1)
    eta_str = job.get_progress_eta_str()
    assert "33.33%" in eta_str
    assert job.estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S") in eta_str

    # Completed state
    job.status = "completed"
    job.update_progress(3)
    eta_str = job.get_progress_eta_str()
    assert "100.00%" in eta_str
    assert "Completed" in eta_str

    # Zero items case using fixture
    assert "N/A (no items)" in zero_item_batch_job.get_progress_eta_str()
