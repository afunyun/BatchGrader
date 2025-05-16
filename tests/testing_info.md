# BatchGrader Concurrent Processing: Manual Test Plan

1. Single-Batch (Legacy) Mode
   Test: Run with a small CSV/JSONL file and default config (no forced chunking, token limit not exceeded).
   Expect: System runs in legacy mode, processes file as a single batch, output matches previous behavior.
   -- SUCCESS --
   Processed normally, output as expected, all logged.

2. Forced Chunking
   Test: Set force_chunk_count to 2 or 3 in config, use an input file with >10 rows.
   Expect: File is split into N chunks, each chunk is processed as a separate batch job, results are aggregated. Output row count matches input.
   -- SUCCESS --
   Chunked properly and processed normally

3. Token-Based Chunking
   Test: Set split_token_limit low (e.g., 1000), use a file that will be split into multiple chunks by token count.
   Expect: File is split by token limit, all chunks processed concurrently (up to max_simultaneous_batches), results are aggregated.
   -- SUCCESS --
   Chunked to like 20 chunks of like 50 tokens each, processed normally.

4. Concurrency Limit
   Test: Set max_simultaneous_batches to 1 and to >1, use a file that will be split into multiple chunks.
   Expect: Only N jobs run in parallel, others wait their turn. (Monitor logs for parallelism.)
   -- SUCCESS --
   Only one job ran at a time despite there being 3 jobs total, obeyed config properly.

5. Halt on Chunk Failure
   Test: Set halt_on_chunk_failure to True, intentionally cause a chunk to fail (e.g., corrupt a row or use a prompt that triggers an API error).
   Expect: On first chunk failure, remaining jobs are halted/cancelled, error is logged, partial results are saved if possible.
   -- SUCCESS --
   Stopp properly when receiving the failure chunk

6. All Chunks Fail
   Test: Use an input that will cause all chunks to fail (e.g., all rows are invalid).
   Expect: System logs batch failure, does not crash, and does not produce a misleading output file.
   -- SUCCESS --
   System logged batch failure, skipped output file creation, and exited gracefully.

7. Empty Input File
   Test: Use an empty file as input.
   Expect: System logs and skips processing, does not crash.
   -- SUCCESS --
   Empty input is detected and skipped without errors.

8. Large Input File
   Test: Use a large file (e.g., >100,000 rows), with chunking enabled.
   Expect: Only the first 50,000 rows are processed (per batch limit), chunking and aggregation work, no crash.
   -- SUCCESS --
   Processed large file in chunks, respected batch size limit, and aggregated results successfully.

9. Backward Compatibility
   Test: Run with old config (no new keys), ensure system falls back to defaults and still works.
   -- SUCCESS --
   Default behaviors applied correctly; legacy configurations are supported.

10. Logging and Output
    Test: For all above, verify logs (console and scratchpad) are clear, errors are descriptive, and output files are correct.
    -- SUCCESS --
    Logs and outputs verified for clarity, correctness, and proper file naming.
