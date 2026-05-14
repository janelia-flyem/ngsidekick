annotations.precomputed
=======================

Export annotations in neuroglancer's `precomputed annotations format
<https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.
The single entry point is
:func:`~ngsidekick.annotations.precomputed.write_precomputed_annotations`,
which supports five annotation types: ``'point'``, ``'line'``,
``'axis_aligned_bounding_box'``, ``'ellipsoid'``, and ``'polyline'``.


Geometry columns
----------------

Annotations can live in a coordinate space of any dimensionality (not just 3D);
the column names are derived from ``coord_space.names``. For example, with
``coord_space.names == ['x', 'y', 'z']``, the input table must have the
following geometry columns (plus any property / relationship columns):

- ``'point'``: ``x``, ``y``, ``z``
- ``'line'`` and ``'axis_aligned_bounding_box'``: ``xa``, ``ya``, ``za``, ``xb``, ``yb``, ``zb``
- ``'ellipsoid'``: ``x``, ``y``, ``z``, ``rx``, ``ry``, ``rz``
- ``'polyline'``: no geometry columns; vertices are supplied separately via ``polyline_points``

For a pandas DataFrame input the index supplies the annotation ID; for a
Feather file input the ID is resolved from (in order) an explicit
``annotation_id`` column, a pandas-index column carried in the file's
schema metadata, or finally synthesized as ``0, 1, 2, ...`` if neither is
available -- so a file written via ``df.to_feather(path)`` just works.
See `Streaming from a Feather file`_ below for the full resolution order.
In most use-cases the annotation ID is not user-visible, so the values
need not be carefully chosen; any unique uint64-compatible values will
do (e.g. ``range(len(df))``).


Examples
--------

Point annotations
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from ngsidekick.annotations.precomputed import write_precomputed_annotations

    df = pd.DataFrame({
        'x': [10.0, 20.0, 30.0],
        'y': [10.0, 20.0, 30.0],
        'z': [10.0, 20.0, 30.0],
    })

    # df:
    #       x     y     z
    #    0  10.0  10.0  10.0
    #    1  20.0  20.0  20.0
    #    2  30.0  30.0  30.0

    write_precomputed_annotations(df, 'xyz', 'point', output_dir='out/points')


Line annotations
^^^^^^^^^^^^^^^^

.. code-block:: python

    df = pd.DataFrame({
        'xa': [0.0, 10.0], 'ya': [0.0, 0.0], 'za': [0.0, 0.0],
        'xb': [5.0, 15.0], 'yb': [5.0, 5.0], 'zb': [0.0, 0.0],
    })

    # df:
    #         xa   ya   za    xb   yb   zb
    #    0   0.0  0.0  0.0   5.0  5.0  0.0
    #    1  10.0  0.0  0.0  15.0  5.0  0.0

    write_precomputed_annotations(df, 'xyz', 'line', output_dir='out/lines')

Bounding boxes use the same column convention as lines, with ``annotation_type='axis_aligned_bounding_box'``.


Ellipsoid annotations
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    df = pd.DataFrame({
        'x':  [10.0, 20.0], 'y':  [10.0, 20.0], 'z':  [10.0, 20.0],
        'rx': [ 2.0,  3.0], 'ry': [ 2.0,  3.0], 'rz': [ 2.0,  3.0],
    })

    # df:
    #          x     y     z   rx   ry   rz
    #    0  10.0  10.0  10.0  2.0  2.0  2.0
    #    1  20.0  20.0  20.0  3.0  3.0  3.0

    write_precomputed_annotations(df, 'xyz', 'ellipsoid', output_dir='out/ellipsoids')


Polyline annotations
^^^^^^^^^^^^^^^^^^^^

Polylines have a variable number of vertices, so vertex coordinates are passed
in a separate auxiliary DataFrame supplied via the ``polyline_points``
argument: one row per vertex, with coordinate columns plus an
``'annotation_id'`` column linking each vertex back to its polyline. Vertex
order within an annotation defines the polyline's traversal order.

The main DataFrame carries any per-annotation properties or relationships;
its index supplies the annotation IDs referenced by ``polyline_points['annotation_id']``.
In the example below, two polylines each get a single ``mycolor`` rgb color
property; the main DataFrame's default ``RangeIndex`` ``[0, 1]`` matches the
``annotation_id`` values in the points table.

.. code-block:: python

    main_df = pd.DataFrame({
        'mycolor_r': [255,   0],
        'mycolor_g': [128, 200],
        'mycolor_b': [  0, 255],
    })

    # main_df:
    #       mycolor_r  mycolor_g  mycolor_b
    #    0        255        128          0
    #    1          0        200        255

    polyline_points = pd.DataFrame({
        'x':             [0.0, 1.0, 2.0,    5.0, 5.0],
        'y':             [0.0, 0.5, 1.0,    5.0, 6.0],
        'z':             [0.0, 0.0, 0.0,    0.0, 0.0],
        'annotation_id': [   0,   0,   0,     1,   1],
    })

    # polyline_points:
    #         x    y    z  annotation_id
    #    0  0.0  0.0  0.0              0
    #    1  1.0  0.5  0.0              0
    #    2  2.0  1.0  0.0              0
    #    3  5.0  5.0  0.0              1
    #    4  5.0  6.0  0.0              1

    write_precomputed_annotations(
        main_df, 'xyz', 'polyline',
        properties=['mycolor'],
        polyline_points=polyline_points,
        output_dir='out/polylines',
    )

See `Properties and relationships`_ below for the full set of supported
property and relationship column conventions, which apply identically to
polyline annotations.

If your polylines have no properties or relationships, you can omit the main
DataFrame entirely by passing ``None`` as the first argument; the main table
is synthesized from the unique annotation IDs in ``polyline_points``:

.. code-block:: python

    write_precomputed_annotations(
        None, 'xyz', 'polyline',
        polyline_points=polyline_points,
        output_dir='out/polylines',
    )


Properties and relationships
----------------------------

In addition to geometry columns, the main DataFrame can carry annotation
**properties** (per-annotation attributes like color or a confidence score)
and **relationships** (per-annotation lists of related segment IDs that
neuroglancer can use to filter annotations by segment).

- **Numeric properties** are plain numeric columns. The column dtype determines
  the encoded type (``uint8``, ``int8``, ..., ``float32``).
- **Enum properties** are pandas categorical columns. Each category becomes a
  discrete enum value with the category label shown in the neuroglancer UI.
- **Color properties** (``rgb`` or ``rgba``) are split across one column per
  channel: ``<name>_r``, ``<name>_g``, ``<name>_b`` (and optionally
  ``<name>_a``). List the *base* name in ``properties``; the suffixed columns
  are picked up automatically.
- **Relationships** are columns whose values are lists of related segment IDs
  (``uint64``). As a shortcut, if every annotation has exactly one related
  segment, the column may have ``dtype=np.uint64`` (a scalar per row) instead
  of containing lists.

The example below demonstrates all four on ``'line'`` annotations. The two
single-segment relationships (``body_pre`` / ``body_post``) use scalar
``uint64`` columns; the multi-segment relationship (``nearby_mito``) uses lists.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from ngsidekick.annotations.precomputed import write_precomputed_annotations

    df = pd.DataFrame({
        # line geometry columns
        'xa': [0.0, 10.0], 'ya': [0.0, 0.0], 'za': [0.0, 0.0],
        'xb': [5.0, 15.0], 'yb': [5.0, 5.0], 'zb': [0.0, 0.0],
        
        # numeric property
        'confidence': [0.92, 0.71],
        
        # enum property (pandas categorical)
        'kind': pd.Categorical(['excitatory', 'inhibitory']),
        
        # color property: one column per channel, rgb(a)
        'mycolor_r': [255,   0], 'mycolor_g': [128, 200], 'mycolor_b': [  0, 255],
        'mycolor_a': [255, 255],  # (alpha is optional)
        
        # single-segment relationships: scalar uint64 per row
        'body_pre':  np.array([100, 200], dtype=np.uint64),
        'body_post': np.array([300, 400], dtype=np.uint64),
        
        # multi-segment relationship: list of uint64 per row
        'nearby_mito': [[10, 11], [20, 21, 22]],
    })

    # df:
    #         xa   ya   za    xb   yb   zb  confidence        kind  mycolor_r  mycolor_g  mycolor_b  mycolor_a  body_pre  body_post   nearby_mito
    #    0   0.0  0.0  0.0   5.0  5.0  0.0        0.92  excitatory        255        128          0        255       100        300      [10, 11]
    #    1  10.0  0.0  0.0  15.0  5.0  0.0        0.71  inhibitory          0        200        255        255       200        400  [20, 21, 22]

    write_precomputed_annotations(
        df, 'xyz', 'line',
        # 'mycolor' is the base name; the _r/_g/_b/_a columns are picked up automatically.
        properties=['confidence', 'kind', 'mycolor'],
        relationships=['body_pre', 'body_post', 'nearby_mito'],
        output_dir='out/lines',
    )


Streaming from a Feather file
-----------------------------

For datasets that don't fit comfortably in pandas memory, ``df`` can be a
path (``str`` or ``os.PathLike``) to a Feather/Arrow IPC file. The file is
memory-mapped via PyArrow and registered into DuckDB; the writers stream
through it in shard-aligned batches so the full input never materializes
in the Python heap.

.. code-block:: python

    import pyarrow.feather as feather

    # Pre-built annotation table on disk -- one row per annotation, with
    # an explicit ``annotation_id`` column (Feather files don't carry a
    # pandas index).
    feather.write_feather(df, 'annotations.feather')

    write_precomputed_annotations(
        'annotations.feather', 'xyz', 'line',
        properties=['confidence'],
        relationships=['body_pre'],
        output_dir='out/lines',
    )

Annotation IDs in Feather files are resolved in priority order:

1. If the file has a column named ``annotation_id``, it is used directly.
2. If the file was written by pandas with a real index column (named or
   anonymous), that column is reused as ``annotation_id`` via a
   zero-copy PyArrow rename. So if you wrote your file with something
   like ``df.to_feather(path)``, the index comes through automatically
   no matter what it was called.
3. If neither applies (e.g. the file was written from a pandas
   ``RangeIndex``, or by a non-pandas tool with no id column), an
   ``annotation_id`` is synthesized on the fly via DuckDB's
   ``ROW_NUMBER()``. A pandas ``RangeIndex(start, stop, step)``
   descriptor in the file's schema metadata, if present, is honored;
   otherwise IDs are ``0, 1, 2, ...``.

Other notes specific to the Feather path:

- Polyline auxiliary data (``polyline_points``) is still required to be
  an in-memory pandas DataFrame. The main table may be Feather even for
  polyline annotations; only the aux table is restricted.
- DuckDB's documented `order-preservation guarantee
  <https://duckdb.org/docs/current/sql/dialect/order_preservation>`_
  ensures that the streamed batches deliver rows in the file's storage
  order, which matters when ``shuffle_before_assigning_spatial_levels=False``.


Tuning tensorstore writes
-------------------------

The sharded write path uses `tensorstore
<https://google.github.io/tensorstore/>`_, which can be tuned via three
related arguments. The defaults are tuned for high-throughput multi-core
machines and should be fine for most cases.

``max_threads`` (default: ``LSB_DJOB_NUMPROC`` on LSF, otherwise the local
CPU count) sets the limit on tensorstore's ``data_copy_concurrency`` and
``file_io_concurrency`` pools — i.e. how many threads tensorstore is
allowed to use for shard encoding/compression and file I/O.

``max_shards_per_transaction`` (default: equal to ``max_threads``) controls
how many shards are committed in a single tensorstore transaction. A
transaction holds all of its shards' staged data in memory until commit,
so this is the main knob for trading **RAM for throughput** during writes:
more shards per transaction → more parallelism at commit time but a higher
peak RAM during sharded writes; fewer shards per transaction → less RAM,
slower commits.

.. code-block:: python

    # Lower memory pressure at the cost of less commit parallelism.
    write_precomputed_annotations(
        df, 'xyz', 'line', output_dir='out/lines',
        max_threads=64,
        max_shards_per_transaction=16,
    )

``tensorstore_context`` accepts a JSON-shaped ``dict`` matching
tensorstore's
`Context spec <https://google.github.io/tensorstore/context.html>`_,
which is useful when you want finer control over tensorstore's resource
pools than ``max_threads`` alone provides. The most useful key in
practice is ``cache_pool.total_bytes_limit``, which caps the
in-memory shard staging that tensorstore retains across transactions
(and tends to dominate sustained RAM use on very large runs):

.. code-block:: python

    write_precomputed_annotations(
        df, 'xyz', 'line', output_dir='out/lines',
        tensorstore_context={
            # Cap tensorstore's internal cache + write-staging pool at 4 GB.
            'cache_pool': {'total_bytes_limit': 4_000_000_000},
        },
    )

Any keys you provide are passed through verbatim; the
``data_copy_concurrency`` and ``file_io_concurrency`` keys are filled
in from ``max_threads`` only when your dict doesn't already specify
them, so you can override one without touching the other.


API reference
-------------

.. automodule:: ngsidekick.annotations.precomputed
   :members:
   :undoc-members:
   :show-inheritance:
