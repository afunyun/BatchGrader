"""
Unit tests for the rich_display module.
"""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.table import Table

from batchgrader.rich_display import RichJobTable, print_summary_table


class MockBatchJob:
    """Mock BatchJob class for testing."""

    def __init__(
        self,
        name="test_job",
        chunk_id_str="chunk_1",
        status="pending",
        openai_batch_id="batch_123",
        error_message=None,
        input_tokens=1000,
        output_tokens=500,
        cost=0.015,
    ):
        self.name = name
        self.chunk_id_str = chunk_id_str
        self.status = status
        self.openai_batch_id = openai_batch_id
        self.error_message = error_message
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost


@pytest.fixture
def mock_console():
    """Create a mock console for testing."""
    return MagicMock(spec=Console)


@pytest.fixture
def mock_jobs():
    """Create a list of mock jobs with different statuses."""
    return [
        MockBatchJob(name="job1", chunk_id_str="chunk_1", status="pending"),
        MockBatchJob(
            name="job2",
            chunk_id_str="chunk_2",
            status="submitted",
            openai_batch_id="batch_456",
        ),
        MockBatchJob(
            name="job3",
            chunk_id_str="chunk_3",
            status="in_progress",
            openai_batch_id="batch_789",
        ),
        MockBatchJob(
            name="job4",
            chunk_id_str="chunk_4",
            status="completed",
            openai_batch_id="batch_101",
        ),
        MockBatchJob(
            name="job5",
            chunk_id_str="chunk_5",
            status="failed",
            openai_batch_id="batch_202",
            error_message="API error",
        ),
    ]


def test_rich_job_table_init():
    """Test initializing the RichJobTable class."""
    # Test with default console
    table = RichJobTable()
    assert isinstance(table.console, Console)

    # Test with provided console
    mock_console = MagicMock()
    table = RichJobTable(console=mock_console)
    assert table.console == mock_console


def test_build_table(mock_jobs):
    """Test building a table from job data."""
    table = RichJobTable()
    result = table.build_table(mock_jobs)

    # Check the result is a table
    assert isinstance(result, Table)

    # Check table title and structure
    assert result.title == "BatchGrader Job Status"
    assert len(result.columns) == 5  # Should have 5 columns


def test_build_table_status_formatting(mock_jobs):
    """Test that job statuses are properly formatted in the table."""
    with patch("rich.table.Table.add_row") as mock_add_row:
        table = RichJobTable()
        table.build_table(mock_jobs)

        # Check colors and emojis for different statuses
        calls = mock_add_row.call_args_list

        # First job (pending)
        assert "⏳" in calls[0][0][2]  # Status column
        assert "[yellow]PENDING[/yellow]" in calls[0][0][2]

        # Second job (submitted)
        assert "📤" in calls[1][0][2]
        assert "[cyan]SUBMITTED[/cyan]" in calls[1][0][2]

        # Third job (in_progress)
        assert "🔄" in calls[2][0][2]
        assert "[cyan]IN_PROGRESS[/cyan]" in calls[2][0][2]

        # Fourth job (completed)
        assert "✅" in calls[3][0][2]
        assert "[green]COMPLETED[/green]" in calls[3][0][2]

        # Fifth job (failed)
        assert "❌" in calls[4][0][2]
        assert "[red]FAILED[/red]" in calls[4][0][2]
        assert "API error" in calls[4][0][4]  # Error message column


def test_build_table_progress_formatting(mock_jobs):
    """Test that progress indicators are properly formatted in the table."""
    with patch("rich.table.Table.add_row") as mock_add_row:
        table = RichJobTable()
        table.build_table(mock_jobs)

        calls = mock_add_row.call_args_list

        # First job (pending) - 0%
        assert "[yellow]░ 0%" in calls[0][0][3]

        # Second job (submitted) - progress indicator
        assert "[cyan]▒ ..." in calls[1][0][3]

        # Third job (in_progress) - progress indicator
        assert "[cyan]▒ ..." in calls[2][0][3]

        # Fourth job (completed) - 100%
        assert "[green]█ 100%" in calls[3][0][3]

        # Fifth job (failed) - 0%
        assert "[red]█ 0%" in calls[4][0][3]


@patch("rich.console.Console.print")
def test_print_summary_table(mock_print, mock_jobs):
    """Test printing a summary table with job statistics."""
    print_summary_table(mock_jobs)

    # Check that Console.print was called with a Table
    mock_print.assert_called_once()
    table_arg = mock_print.call_args[0][0]
    assert isinstance(table_arg, Table)
    assert table_arg.title == "BatchGrader Job Summary"

    # Can't easily check the exact table content without more mocking


@patch("rich.table.Table.add_row")
def test_print_summary_table_content(mock_add_row, mock_jobs):
    """Test the content of the summary table."""
    with patch("rich.console.Console.print"):
        print_summary_table(mock_jobs)

    # Should have been called once with the computed statistics
    mock_add_row.assert_called_once()

    # Parse the arguments to verify values
    args = mock_add_row.call_args[0]
    assert args[0] == "5"  # Total jobs
    assert args[1] == "1"  # Succeeded (completed)
    assert args[2] == "1"  # Failed
    assert args[3] == "0"  # Errored
    assert args[4] == "7500"  # Total tokens (input + output)
    assert args[5] == "0.0750"  # Total cost (formatted with 4 decimal places)
