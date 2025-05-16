"""
BatchJob abstraction for concurrent batch processing in BatchGrader.

This class encapsulates the state, metadata, and results for a single chunk/job in the concurrent batch system.
It is used by the batch runner to track submission, polling, completion, and error state for each chunked batch.

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

from typing import Any, Dict, Optional, Union
import os
import time
import datetime

import pandas as pd


class BatchJob:

    def __init__(self,
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
                 result_data: Optional[Union[pd.DataFrame, dict]] = None):

        self.chunk_id_str = chunk_id_str
        self.chunk_df = chunk_df
        self.system_prompt = system_prompt
        self.response_field = response_field
        self.original_filepath = original_filepath
        self.chunk_file_path = chunk_file_path
        self.llm_model = llm_model
        self.api_key_prefix = api_key_prefix

        self.status = status
        self.error_message = error_message
        self.error_details = error_details
        self.result_data = result_data

        self.openai_batch_id: Optional[str] = None
        self.input_file_id_for_chunk: Optional[str] = None

        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost: float = 0.0

        # Progress tracking attributes
        self.total_items: int = len(chunk_df) if chunk_df is not None else 0
        self.processed_items: int = 0
        self.start_time: Optional[float] = None
        self.estimated_completion_time: Optional[datetime.datetime] = None

    def update_progress(self, items_processed_increment: int):
        """Updates the progress of the job."""
        if self.start_time is None:
            self.start_time = time.monotonic()

        self.processed_items += items_processed_increment
        self.processed_items = min(self.processed_items,
                                   self.total_items)  # Cap at total_items

        if self.processed_items > 0 and self.total_items > 0 and self.start_time is not None:
            elapsed_time = time.monotonic() - self.start_time
            time_per_item = elapsed_time / self.processed_items
            remaining_items = self.total_items - self.processed_items
            remaining_time_seconds = remaining_items * time_per_item
            self.estimated_completion_time = datetime.datetime.now(
            ) + datetime.timedelta(seconds=remaining_time_seconds)
        elif self.processed_items == self.total_items:  # Job completed
            self.estimated_completion_time = datetime.datetime.now()

    def get_progress_eta_str(self) -> str:
        """Returns a string with progress percentage and ETA."""
        if self.total_items == 0:
            return "N/A (no items)"

        progress_percent = (self.processed_items / self.total_items) * 100

        eta_str = "Calculating..."
        if self.status == "completed":
            eta_str = "Completed"
        elif self.estimated_completion_time:
            eta_str = self.estimated_completion_time.strftime(
                "%Y-%m-%d %H:%M:%S")
        elif self.status == "running" and self.processed_items == 0:
            eta_str = "Started, calculating ETA..."

        return f"{progress_percent:.2f}% (ETA: {eta_str})"

    def get_status_log_str(self) -> str:
        """Returns a formatted string for logging this job's current state."""
        base = f"Chunk {self.chunk_id_str} of {os.path.basename(self.original_filepath)}: {self.status}"
        if self.openai_batch_id:
            base += f" (Batch ID: {self.openai_batch_id})"
        if self.error_message:
            base += f" [ERROR: {self.error_message}]"
        return base
