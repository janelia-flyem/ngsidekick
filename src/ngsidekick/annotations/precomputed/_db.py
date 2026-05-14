"""
DuckDB orchestration for the precomputed-annotations writers.

The writers that use this path receive a DuckDB connection (rather than
a pandas DataFrame) and query the input table in shard-aligned batches,
encoding each batch on the fly. This caps peak RAM during the
encode+write phase at roughly one batch's worth of data rather than the
full dataset.

The input is exposed to DuckDB as the view :data:`INPUT_VIEW`, sourced
from one of:

- a pandas DataFrame the user passed in (registered as a zero-copy
  Arrow view; data lives in pandas' heap),
- (future) a Feather/Arrow IPC file path (memory-mapped by DuckDB;
  data lives in the kernel page cache rather than in the Python heap).
"""
import os
from typing import Union

import pandas as pd
import duckdb

INPUT_VIEW = 'annotations_input'


def open_connection(memory_limit: str | None = None, threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory DuckDB connection with optional resource limits.

    ``memory_limit`` is forwarded to DuckDB's ``memory_limit`` setting
    (e.g. ``'16GB'``). When None, DuckDB defaults to ~80% of physical
    RAM. ``threads`` caps the number of worker threads DuckDB uses for
    query execution; defaults to CPU count.
    """
    con = duckdb.connect()  # in-memory database
    if memory_limit is not None:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if threads is not None:
        con.execute(f"SET threads = {threads}")
    return con


def register_input(
    con: duckdb.DuckDBPyConnection,
    df_or_path: Union[pd.DataFrame, str, os.PathLike],
) -> None:
    """
    Register the user's annotation table as the DuckDB view :data:`INPUT_VIEW`.

    For pandas inputs, the DataFrame's index is materialized as an
    ``annotation_id`` column (zero-copy for typical uint64 indexes — the
    column shares its buffer with the index).
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
        if 'annotation_id' not in df.columns:
            df = df.copy(deep=False)
            df.insert(0, 'annotation_id', df.index.to_numpy())
        con.register(INPUT_VIEW, df)
    elif isinstance(df_or_path, (str, os.PathLike)):
        # Feather/Arrow IPC path. Not yet wired up to a public API but
        # supported here so the writers don't have to branch on input
        # shape.
        path = os.fspath(df_or_path)
        con.execute(f"CREATE VIEW {INPUT_VIEW} AS SELECT * FROM read_ipc('{path}')")
    else:
        raise TypeError(
            f"Expected pandas DataFrame or path-like, got {type(df_or_path).__name__}"
        )
