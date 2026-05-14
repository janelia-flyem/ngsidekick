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
- a Feather/Arrow IPC file path (memory-mapped by DuckDB; data lives
  in the kernel page cache rather than in the Python heap).
"""
import os
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather
import duckdb

INPUT_VIEW = 'annotations_input'

# Internal helper handle used when we need to wrap a registered source
# with an additional view (e.g. for polyline orphan filtering).
_INPUT_VIEW_RAW = '_annotations_input_raw'
_FILTER_ANN_IDS = '_annotations_input_filter_ids'


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

    For path-like inputs, the file is read via DuckDB's ``read_ipc``.
    The file must contain an ``annotation_id`` column.

    A raw view is stashed under an internal name so callers can later
    swap INPUT_VIEW for a filtered version via
    :func:`restrict_input_to_ids` without losing the original source.
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
        if 'annotation_id' not in df.columns:
            df = df.copy(deep=False)
            df.insert(0, 'annotation_id', df.index.to_numpy())
        con.register(_INPUT_VIEW_RAW, df)
    elif isinstance(df_or_path, (str, os.PathLike)):
        # Memory-map the file via PyArrow and register the resulting
        # Arrow table. The kernel handles demand paging, so a multi-GB
        # file doesn't get fully resident in RAM -- DuckDB scans through
        # touched pages and lets the rest stay on disk.
        path = os.fspath(df_or_path)
        arrow_table = pa.feather.read_table(path, memory_map=True)
        con.register(_INPUT_VIEW_RAW, arrow_table)
    else:
        raise TypeError(
            f"Expected pandas DataFrame or path-like, got {type(df_or_path).__name__}"
        )
    con.execute(f"CREATE OR REPLACE VIEW {INPUT_VIEW} AS SELECT * FROM {_INPUT_VIEW_RAW}")


def restrict_input_to_ids(
    con: duckdb.DuckDBPyConnection,
    valid_annotation_ids: np.ndarray,
) -> None:
    """
    Replace :data:`INPUT_VIEW` with a JOIN-filtered view that only
    includes rows whose ``annotation_id`` is in ``valid_annotation_ids``.

    Used by the polyline path to drop main-table rows that have no
    matching vertices in the polyline aux table. Matches the
    pandas-side behavior of ``df.loc[valid_mask]``.
    """
    valid_table = pa.table({
        'annotation_id': np.asarray(valid_annotation_ids, dtype=np.uint64),
    })
    con.register(_FILTER_ANN_IDS, valid_table)
    con.execute(f"""
        CREATE OR REPLACE VIEW {INPUT_VIEW} AS
        SELECT r.* FROM {_INPUT_VIEW_RAW} r
        JOIN {_FILTER_ANN_IDS} v ON v.annotation_id = r.annotation_id
    """)
