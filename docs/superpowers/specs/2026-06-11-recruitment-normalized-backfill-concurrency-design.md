# Recruitment Normalized Backfill Concurrency Design

**Goal**

Upgrade `src.data_pipeline.backfill_recruitment_jobs_normalized` so it can safely process static source tables at 10M+ row scale with chunked SQL-side backfill, configurable worker concurrency, resumable execution, and benchmark-friendly logging.

## Scope

This design covers:

- `src/data_pipeline/backfill_recruitment_jobs_normalized.py`
- PostgreSQL engine reuse and pool configuration in `src/db/postgres.py`
- SQL-side chunk execution helpers around `public.recruitment_jobs_normalized`
- Benchmark and run-state logging needed to tune `workers × chunk_size`

This design does not cover:

- Changing the source-table schemas
- Replacing PostgreSQL with another execution engine
- Rewriting the historical DataFrame fallback into a separate pipeline

## Confirmed Assumptions

- The three source tables are treated as static read-only inputs during a run.
- `row_number() over (order by ctid)` is acceptable for building one run-local stable locator snapshot.
- PostgreSQL remains the only formal storage layer for structured data.

## Current-State Findings

The current implementation has two paths:

- A default SQL bulk path that processes each source table serially with one large `INSERT ... SELECT`.
- A Python DataFrame fallback path that loops in batches and upserts row by row.

This creates four bottlenecks for 10M+ rows:

1. No script-level concurrency control.
2. Repeated engine creation and disposal instead of pooled reuse.
3. Whole-table SQL work that cannot be retried at chunk granularity.
4. No benchmark output that makes tuning easy.

## Recommended Approach

Use database-side chunk execution with Python thread-pool scheduling.

The script should:

1. Create one run-scoped locator snapshot per source table.
2. Split locator row numbers into deterministic chunks.
3. Dispatch chunk jobs to a configurable worker pool.
4. Execute each chunk in its own transaction through SQL-only `INSERT ... SELECT`.
5. Persist chunk-level run state and benchmark metrics.

This keeps heavy filtering, joining, and insertion inside PostgreSQL while letting Python coordinate parallelism and reporting.

## Alternatives Considered

### Option A: SQL-side chunk concurrency with Python worker scheduling

This is the recommended approach. It minimizes Python-side memory pressure, supports partial retries, and stays close to the current SQL bulk design.

### Option B: Parallelize the pandas batch path

This would be simpler mechanically but performs worse at 10M+ scale because it moves too much data through Python and multiplies DataFrame overhead.

### Option C: Rely only on PostgreSQL internal parallelism

This reduces application complexity but does not provide explicit `--workers`, chunk retries, or benchmark controls, which are part of the requested outcome.

## Execution Model

The CLI should remain a single entrypoint but add these controls:

- `--workers`
- `--chunk-size`
- `--max-chunks`
- `--benchmark`
- `--benchmark-json`
- `--resume-run-id`
- `--retry-failed-chunks`
- `--max-retries`
- `--db-pool-size`
- `--db-max-overflow`

Default behavior should keep the SQL bulk path as the preferred mode. The pandas path remains available only as a compatibility fallback.

Each chunk should be a transaction boundary. Failures should be isolated to one chunk, not one table or one full run.

## Database Design

### Locator snapshot

Add a run-scoped locator table in `public` that stores:

- `run_id`
- `source_table`
- `source_row_number`
- `source_ctid`
- `chunk_id`
- `created_at`

The locator table should be filled once per source table using `row_number() over (order by ctid)`. This converts repeated whole-table numbering work into one upfront sequential pass.

### Run-state table

Add a lightweight run-state table in `public` for chunk orchestration:

- `run_id`
- `source_table`
- `chunk_id`
- `range_start`
- `range_end`
- `planned_rows`
- `attempt`
- `status`
- `started_at`
- `finished_at`
- `duration_seconds`
- `written_rows`
- `error_message`

Statuses should include `pending`, `running`, `succeeded`, `failed`, and `skipped`.

### Chunk SQL

Each worker should:

1. Read locator rows for one `(source_table, chunk_id)` range.
2. Join back to the source table through `ctid`.
3. Join to the normalized table by `(source_table, source_row_number)` when `only_missing=True`.
4. Insert normalized rows through one SQL statement.

When `only_missing=True`, the statement should prefer `ON CONFLICT DO NOTHING`.

When `all_source_rows=True`, the statement should use `ON CONFLICT ... DO UPDATE` with `IS DISTINCT FROM` guards so unchanged rows are not rewritten.

### Indexes and cleanup

The normalized target must keep its unique source-locator index.

The locator table should have an index on `(run_id, source_table, chunk_id)` and `(run_id, source_table, source_row_number)`.

The run-state table should have an index on `(run_id, status)` and a uniqueness constraint on `(run_id, source_table, chunk_id, attempt)`.

Locator cleanup should be explicit so old runs do not accumulate indefinitely.

## Engine and Pooling

`src/db/postgres.py` should allow pooled engine tuning:

- cache resolved database name
- accept `pool_size`
- accept `max_overflow`
- enable `pool_pre_ping`
- allow `pool_recycle`
- allow `application_name`

One engine should be created for the run and reused by all worker threads.

## Benchmark and Logging

Two logging levels are required:

### Chunk log

One structured record per chunk:

- `run_id`
- `source_table`
- `chunk_id`
- `range_start`
- `range_end`
- `planned_rows`
- `written_rows`
- `duration_seconds`
- `rows_per_second`
- `worker_name`
- `attempt`
- `status`
- `error_message`

### Run summary

One summary per run with:

- total duration
- total planned rows
- total written rows
- global rows per second
- per-table throughput
- chunk duration `p50`, `p95`, `max`
- failure count
- retry count
- effective worker and pool settings

Console output should stay readable. Detailed benchmark output should be written to JSON or JSONL when requested.

## Failure Handling

Chunk failures should retry automatically up to `--max-retries`.

The default run policy should continue other chunks after one chunk fails, then exit non-zero if any chunk remains failed.

`--resume-run-id` should reuse existing locator and run-state metadata and only schedule unfinished or failed chunks.

## Validation Strategy

Validation should cover:

- unit tests for chunk planning and summary aggregation
- unit tests for benchmark metric calculation
- parser tests for new CLI flags
- compilation of `src`

Runtime validation should include a smoke benchmark on a limited number of chunks before full runs.

## Initial Tuning Guidance

Benchmarking on the current machine and PostgreSQL settings showed the most stable default is:

- `workers=3`
- `chunk_size=50000`
- `db_pool_size=3`
- `db_max_overflow=3`

Observed benchmark findings:

- `200000` row chunks were too large and produced heavy long-tail behavior.
- `4 × 100000` was slower than `3 × 100000`, which indicates PostgreSQL write pressure saturated before CPU.
- `3 × 50000` completed the sampled `51job` chunks in roughly `40` to `50` seconds each and the sampled `Liepin` chunks in roughly `65` to `74` seconds each, with less long-tail variance.

Future comparisons should stay near the observed stable zone:

- `3 × 50000`
- `2 × 50000`
- `3 × 75000`

The current recommended production default is `3 × 50000`. PostgreSQL write pressure dominates before CPU saturation on this workload.

## Risks and Mitigations

- `ctid` instability is mitigated by the confirmed static-read-only assumption.
- Large locator snapshots add temporary storage cost; explicit cleanup keeps it bounded.
- Too many workers may reduce throughput; benchmark logging is built in to prevent blind scaling.
- Mixed old and new execution modes may drift; the pandas path should remain clearly marked as fallback-only.
