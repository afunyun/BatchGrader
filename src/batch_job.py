"""
BatchJob abstraction for concurrent batch processing in BatchGrader.

This class encapsulates the state, metadata, and results for a single chunk/job in the concurrent batch system.
It is used by the batch runner to track submission, polling, completion, and error state for each chunked batch.

Attributes:
    chunk_data_identifier: Path to the chunk's data file (CSV/JSON/JSONL)
    chunk_df: Loaded pandas DataFrame for the chunk (can be None until loaded)
    system_prompt: The prompt content used for this chunk
    response_field: The column name for LLM output
    original_source_file: The basename of the parent input file
    chunk_id_str: String identifying this chunk (e.g., 'forced_1_of_3', 'split_part_2.csv')
    llm_model: Model name used for this chunk
    api_key_prefix: API key prefix (for logging)
    openai_batch_id: OpenAI's batch job ID for this chunk (after submission)
    status: String status ('pending', 'submitted', 'polling', 'completed', 'failed', 'error', ...)
    results_df: pandas DataFrame holding results for this chunk (if completed)
    error_message: Details of any failure (if applicable)
    input_file_id_for_chunk: OpenAI File ID for the uploaded chunk data
    df_chunk_with_custom_ids: DataFrame chunk with custom_ids added by _prepare_batch_requests
    input_tokens: int, total input tokens used for this chunk
    output_tokens: int, total output tokens used for this chunk
    cost: float, cost for this chunk (populated after completion)

Methods:
    get_status_log_str(): Returns a formatted string for logging this job's current state.
"""

import pandas as pd
from typing import Optional

class BatchJob:
    def __init__(self,
                chunk_data_identifier: str,
                chunk_df: Optional[pd.DataFrame],
                system_prompt: str,
                response_field: str,
                original_source_file: str,
                chunk_id_str: str,
                llm_model: Optional[str] = None,
                api_key_prefix: Optional[str] = None):
        self.chunk_data_identifier = chunk_data_identifier
        self.chunk_df = chunk_df
        self.system_prompt = system_prompt
        self.response_field = response_field
        self.original_source_file = original_source_file
        self.chunk_id_str = chunk_id_str
        self.llm_model = llm_model
        self.api_key_prefix = api_key_prefix
        self.openai_batch_id = None
        self.status = "pending"
        self.results_df = None
        self.error_message = None
        self.input_file_id_for_chunk = None
        self.df_chunk_with_custom_ids = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def get_status_log_str(self) -> str:
        """Returns a formatted string for logging this job's current state."""
        base = f"Chunk {self.chunk_id_str} of {self.original_source_file}: {self.status}"
        if self.openai_batch_id:
            base += f" (Batch ID: {self.openai_batch_id})"
        if self.error_message:
            base += f" [ERROR: {self.error_message}]"
        return base
