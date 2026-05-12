import logging
import numpy as np

from ._encode import _encode_annotation_records, _encode_relationship_records
from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


def _write_annotations_by_id(df, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                              output_dir, write_sharded, max_shards_per_transaction, ts_context):
    """
    Write the annotations to the "Annotation ID Index".

    Each annotation's value is its encoded geometry+property record followed
    by its encoded relationship record. Encoding is performed inline here
    (rather than upstream) so that no per-row byte objects are persisted
    between writers.

    Args:
        df:
            DataFrame holding the native geometry / property / relationship
            columns for every annotation. The index supplies the by-id keys.
        coord_space, annotation_type, property_specs, relationships, polyline_geom:
            See :func:`write_precomputed_annotations`.

        output_dir, write_sharded, max_shards_per_transaction, ts_context:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        JSON metadata for the 'by_id' key in the top-level 'info' file.
    """
    logger.info("Encoding annotations for 'by_id' index")
    ann_pb = _encode_annotation_records(
        df, coord_space, annotation_type, property_specs, polyline_geom,
    )
    rel_pb = _encode_relationship_records(df, relationships)

    buffers = [ann_pb]
    if rel_pb is not None:
        buffers.append(rel_pb)

    logger.info("Writing annotations to 'by_id' index")
    return _write_buffers(
        df.index,
        buffers,
        output_dir,
        "by_id",
        write_sharded,
        max_shards_per_transaction,
        ts_context,
    )
