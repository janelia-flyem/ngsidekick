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

# Internal helper handles. We layer two views over the user's raw input:
#
#   _INPUT_RAW_SOURCE: the registered pandas DataFrame or Arrow table.
#                      May or may not have an ``annotation_id`` column
#                      (Feather files might rely on a pandas-index
#                      column or have no id column at all).
#   _INPUT_VIEW_RAW:   a view over the raw source that is guaranteed to
#                      expose ``annotation_id`` -- either by passthrough,
#                      by renaming a pandas-index column, or by
#                      synthesizing it via ROW_NUMBER().
#   INPUT_VIEW:        the public view callers consume. Usually a
#                      passthrough of _INPUT_VIEW_RAW; may be replaced
#                      with a JOIN-filtered version via
#                      restrict_input_to_ids (polyline orphans).
_INPUT_RAW_SOURCE = '_annotations_input_source'
_INPUT_VIEW_RAW = '_annotations_input_raw'
_FILTER_ANN_IDS = '_annotations_input_filter_ids'


def open_connection(memory_limit: str | None = None, threads: int | None = None,
                    temp_directory: str | None = None) -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory DuckDB connection with optional resource limits.

    ``memory_limit`` is forwarded to DuckDB's ``memory_limit`` setting
    (e.g. ``'16GB'``). When None, DuckDB defaults to ~80% of physical
    RAM. ``threads`` caps the number of worker threads DuckDB uses for
    query execution; defaults to CPU count.

    ``temp_directory`` is forwarded to DuckDB's ``temp_directory``
    setting (the location for spill files when the working set exceeds
    ``memory_limit``). When None, DuckDB defaults to ``.tmp/`` in the
    process's current working directory.
    """
    con = duckdb.connect()  # in-memory database
    if memory_limit is not None:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if threads is not None:
        con.execute(f"SET threads = {threads}")
    if temp_directory is not None:
        # SET temp_directory accepts a quoted path; quote it ourselves
        # to avoid issues with single-quote characters in the value.
        escaped = temp_directory.replace("'", "''")
        con.execute(f"SET temp_directory = '{escaped}'")
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

    For path-like inputs, the Feather/Arrow IPC file is memory-mapped
    via PyArrow and registered as an Arrow table; the kernel handles
    demand paging so multi-GB files don't go fully resident. The
    ``annotation_id`` column is sourced according to
    :func:`_annotation_id_strategy` -- if the file already has an
    ``annotation_id`` column it's used as-is; if the file was written by
    pandas and carries a real index column in its schema metadata, that
    column is renamed to ``annotation_id`` via zero-copy PyArrow rename;
    otherwise ``annotation_id`` is synthesized via a DuckDB view that
    adds ``ROW_NUMBER() OVER ()`` (honoring start/step from a pandas
    RangeIndex descriptor when present).
    """
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
        if 'annotation_id' not in df.columns:
            df = df.copy(deep=False)
            df.insert(0, 'annotation_id', df.index.to_numpy())
        con.register(_INPUT_RAW_SOURCE, df)
        con.execute(f"CREATE OR REPLACE VIEW {_INPUT_VIEW_RAW} AS SELECT * FROM {_INPUT_RAW_SOURCE}")
    elif isinstance(df_or_path, (str, os.PathLike)):
        path = os.fspath(df_or_path)
        arrow_table = pa.feather.read_table(path, memory_map=True)
        strategy, info = _annotation_id_strategy(arrow_table)
        if strategy == 'rename':
            # rename_columns shares the underlying buffers; the new
            # Table object is a thin schema-only wrapper.
            new_names = [
                'annotation_id' if c == info else c
                for c in arrow_table.column_names
            ]
            arrow_table = arrow_table.rename_columns(new_names)
        con.register(_INPUT_RAW_SOURCE, arrow_table)
        if strategy == 'synthesize':
            start, step = info
            con.execute(f"""
                CREATE OR REPLACE VIEW {_INPUT_VIEW_RAW} AS
                SELECT
                    ({start} + (ROW_NUMBER() OVER () - 1) * {step})::BIGINT AS annotation_id,
                    *
                FROM {_INPUT_RAW_SOURCE}
            """)
        else:
            con.execute(f"CREATE OR REPLACE VIEW {_INPUT_VIEW_RAW} AS SELECT * FROM {_INPUT_RAW_SOURCE}")
    else:
        raise TypeError(
            f"Expected pandas DataFrame or path-like, got {type(df_or_path).__name__}"
        )
    con.execute(f"CREATE OR REPLACE VIEW {INPUT_VIEW} AS SELECT * FROM {_INPUT_VIEW_RAW}")


def _annotation_id_strategy(arrow_table):
    """
    Decide how to expose ``annotation_id`` for a Feather-loaded Arrow
    table.

    Priority order:

    1. ``('present', None)`` -- the file already has an
       ``annotation_id`` column. Use as-is. (If both this and a pandas
       index column exist, the explicit column wins.)
    2. ``('rename', col_name)`` -- the file was written by pandas with
       a real index column (either user-named or the synthetic
       ``'__index_level_0__'`` for anonymous indexes). Rename it to
       ``annotation_id``.
    3. ``('synthesize', (start, step))`` -- no usable id column. The
       view layer should add one via ROW_NUMBER. ``start``/``step``
       come from a pandas RangeIndex descriptor when present, else
       ``(0, 1)``.
    """
    if 'annotation_id' in arrow_table.column_names:
        return 'present', None
    pandas_meta = arrow_table.schema.pandas_metadata
    range_info = (0, 1)
    if pandas_meta:
        for entry in pandas_meta.get('index_columns', []):
            if isinstance(entry, str):
                return 'rename', entry
            if isinstance(entry, dict) and entry.get('kind') == 'range':
                range_info = (entry.get('start', 0), entry.get('step', 1))
    return 'synthesize', range_info


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
