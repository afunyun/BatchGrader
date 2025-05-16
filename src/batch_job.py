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

Methods:
    get_status_log_str(): Returns a formatted string for logging this job's current state.
"""

from typing import Any, Dict, Optional, Union
import os

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

    def get_status_log_str(self) -> str:
        """Returns a formatted string for logging this job's current state."""
        base = f"Chunk {self.chunk_id_str} of {os.path.basename(self.original_filepath)}: {self.status}"
        if self.openai_batch_id:
            base += f" (Batch ID: {self.openai_batch_id})"
        if self.error_message:
            base += f" [ERROR: {self.error_message}]"
        return base
