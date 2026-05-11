NGSidekick
==========

Tools for working with `neuroglancer <https://github.com/google/neuroglancer>`_.

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   state_utils
   segmentprops
   annotations
   segmentcolors
   gcs
   ngvideo_helper


Feature Highlights
--------------------

- :doc:`Segment properties <segmentprops>`: read and write neuroglancer's
  precomputed `segment properties
  <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md>`_
  format from a pandas DataFrame.
- :doc:`Annotations <annotations>`:

  - :doc:`Local annotations <local>`: construct local annotations directly
    in viewer state.  (API subject to change.)
  - :doc:`Precomputed annotations <precomputed>`: export annotations in
    neuroglancer's `precomputed annotations format
    <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_
    from a pandas DataFrame.

    - Supports all five annotation types:
        - ``point``
        - ``line``
        - ``axis_aligned_bounding_box``
        - ``ellipsoid``
        - ``polyline``
    - Per-annotation **properties** (numeric, enum/categorical, and rgb/rgba color).
    - Per-annotation **relationships** (lists of related segment IDs, used by
      neuroglancer to filter annotations by segment).
    - Written to all "index" types (annotation id, related segment, and multi-level spatial grid)
    - Sharded output, written in parallel via `tensorstore <https://google.github.io/tensorstore/>`_.
    - **Note:** writing directly to cloud storage is not yet supported;
      outputs must be written to a local filesystem (you must upload to cloud storage afterwards).


Installation
------------

Packages are available from both PyPI and conda-forge.

.. code-block:: bash

   pip install ngsidekick


.. code-block:: bash

   conda install -c conda-forge ngsidekick


For additional features:

.. code-block:: bash

   pip install ngsidekick[gcs]  # For Google Cloud Storage support


.. code-block:: bash

   conda install -c conda-forge ngsidekick google-cloud-storage
