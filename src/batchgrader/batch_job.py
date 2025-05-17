"""
BatchJob abstraction for concurrent batch processing in BatchGrader.

This class encapsulates the state, metadata, and results for a single chunk/job in the concurrent
batch system. It is used by the batch runner to track submission, polling, completion, and error
state for each chunked batch.

Attributes:
    chunk_file_path: Path to the chunk's data file (CSV/JSON/JSONL)
    chunk_df: Loaded pandas DataFrame for the chunk (can be None until loaded)
    system_prompt: The prompt content used for this chunk
    response_field: The column name for LLM output
    original_filepath: The basename of the parent input file
    chunk_id_str: String identifying this chunk (e.g., 'forced_1_of_3', 'split_part_2.csv')
    llm_model: Model name used for this chunk
    api_key_prefix: API key prefix (for logging)
    openai_batch_id: OpenAI's batch job ID for this chunk (after submission)
    status: String status ('pending', 'submitted', 'polling', 'completed', 'failed', 'error', ...)
    error_message: Details of any failure (if applicable)
    error_details: Additional details of any failure (if applicable)
    result_data: pandas DataFrame holding results for this chunk (if completed) or error dict
    input_file_id_for_chunk: OpenAI File ID for the uploaded chunk data
    input_tokens: int, total input tokens used for this chunk
    output_tokens: int, total output tokens used for this chunk
    cost: float, cost for this chunk (populated after completion)
    total_items: int, total number of items in the chunk
    processed_items: int, number of items processed so far
    start_time: Optional[float], timestamp when processing started
    estimated_completion_time: Optional[datetime.datetime], estimated completion time

Methods:
    get_status_log_str(): Returns a formatted string for logging this job's current state.
    update_progress(items_processed_increment: int): Updates the progress of the job.
    get_progress_eta_str(): Returns a string with progress percentage and ETA.
"""

import datetime
import os
import threading
import time
from typing import Optional, Union

import pandas as pd


class BatchJob:
    """
    Represents a single chunk/job in the concurrent batch processing system.

    This class tracks the state, metadata, and results for a batch job, including progress,
    status transitions, and error handling.
    """

    def __init__(
        self,
        chunk_id_str: str,
        chunk_df: Optional[pd.DataFrame],
        system_prompt: str,
        response_field: str,
        original_filepath: str,
        chunk_file_path: str,
        llm_model: Optional[str] = None,
        api_key_prefix: Optional[str] = None,
        status: str = "pending",
        error_message: Optional[str] = None,
        error_details: Optional[str] = None,
        result_data: Optional[Union[pd.DataFrame, dict]] = None,
    ):
        self.chunk_id_str = chunk_id_str
        self.chunk_df = chunk_df
        self.system_prompt = system_prompt
        self.response_field = response_field
        self.original_filepath = original_filepath
        self.chunk_file_path = chunk_file_path
        self.llm_model = llm_model
        self.api_key_prefix = api_key_prefix

        self._status = status
        self.error_message = error_message
        self.error_details = error_details
        self.result_data = result_data

        self.openai_batch_id: Optional[str] = None
        self.input_file_id_for_chunk: Optional[str] = None

        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost: float = 0.0

        self.total_items: int = len(chunk_df) if chunk_df is not None else 0
        self.processed_items: int = 0
        self.start_time: Optional[float] = None
        self.estimated_completion_time: Optional[datetime.datetime] = None

        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, new_status: str):
        allowed_statuses = {
            "pending",
            "submitted",
            "polling",
            "in_progress",
            "running",
            "completed",
            "failed",
            "error",
        }
        old_status = self._status
        if new_status == old_status:
            return
        if new_status not in allowed_statuses:
            raise ValueError(f"Invalid status value: {new_status}")
        if new_status in {
                "error", "failed"
        } or (old_status in {"error", "failed"} and new_status == "completed"):
            self._status = new_status
            return
        order = [
            "pending",
            "submitted",
            "polling",
            "in_progress",
            "running",
            "completed",
        ]
        if (old_status in order and new_status in order
                and order.index(new_status) >= order.index(old_status)):
            self._status = new_status
            return
        raise ValueError(
            f"Invalid state transition from {old_status} to {new_status}")

    def update_progress(self, items_processed_increment: int):
        """
        Update the progress of the batch job.

        Args:
            items_processed_increment: Number of items processed since the last update

        Raises:
            ValueError: If items_processed_increment is not a positive integer
        """
        if items_processed_increment <= 0:
            raise ValueError(
                "items_processed_increment must be a positive integer")

        with self._lock:
            if self.start_time is None and self.total_items > 0:
                self.start_time = time.monotonic()

            self.processed_items = min(
                self.processed_items + items_processed_increment,
                self.total_items)

            if (self.total_items > 0 and self.processed_items > 0
                    and self.start_time is not None):
                elapsed_time = time.monotonic() - self.start_time
                time_per_item = elapsed_time / self.processed_items
                remaining_items = self.total_items - self.processed_items
                remaining_time = remaining_items * time_per_item
                self.estimated_completion_time = (
                    datetime.datetime.now() +
                    datetime.timedelta(seconds=remaining_time)
                    if self.processed_items < self.total_items else
                    datetime.datetime.now())

    def get_progress_eta_str(self) -> str:
        """
        Get a string representing the progress and estimated time of completion.

        Returns:
            str: A formatted string showing progress percentage and estimated completion time
        """
        if self.total_items == 0:
            return "N/A (no items)"

        progress_percent = (self.processed_items / self.total_items) * 100

        if self.processed_items == self.total_items:
            eta_str = "Completed"
        elif self.estimated_completion_time:
            eta_str = self.estimated_completion_time.strftime(
                "%Y-%m-%d %H:%M:%S")
        elif self.processed_items == 0:
            eta_str = "Started, calculating ETA..."
        else:
            eta_str = "Calculating..."

        return f"{progress_percent:.2f}% (ETA: {eta_str})"

    def get_status_log_str(self) -> str:
        """
        Get a string representing the status of the batch job.

        Returns:
            str: A formatted string showing the status of the batch job
        """
        filename = os.path.basename(self.original_filepath)
        base = f"Chunk {self.chunk_id_str} of {filename}: {self.status}"
        if self.openai_batch_id:
            base += f" (Batch ID: {self.openai_batch_id})"
        if self.error_message:
            base += f" [ERROR: {self.error_message}]"
        return base
