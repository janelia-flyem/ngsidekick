import logging
import numpy as np
import pandas as pd

from ._encode import PartitionedBuffer, _encode_annotation_records, _encode_id_bytes
from ._util import _ann_required_cols, PolylineGeometry
from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


def _write_annotations_by_relationships(df, coord_space, annotation_type, property_specs, relationships, polyline_geom,
                                        output_dir, write_sharded, max_shards_per_transaction, max_threads):
    """
    Write the annotations to a "Related Object ID Index" for each relationship.

    Args:
        df:
            DataFrame holding the native geometry / property / relationship
            columns for every annotation. Not mutated.
        coord_space, annotation_type, property_specs, relationships, polyline_geom:
            See :func:`write_precomputed_annotations`.

        output_dir, write_sharded, max_shards_per_transaction, max_threads:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        List of JSON metadata dicts (one per relationship) for the
        'relationships' key in the top-level 'info' file.
    """
    by_rel_metadata = []
    for relationship in relationships:
        metadata = _write_annotations_by_relationship(
            df, coord_space, annotation_type, property_specs, polyline_geom, relationship,
            output_dir, write_sharded, max_shards_per_transaction, max_threads,
        )
        by_rel_metadata.append(metadata)
    return by_rel_metadata


def _write_annotations_by_relationship(df, coord_space, annotation_type, property_specs, polyline_geom, relationship,
                                       output_dir, write_sharded, max_shards_per_transaction, max_threads):
    """
    Write the annotations to a "Related Object ID Index" for a single relationship.

    For each unique related-object id, the output value is::

        <count:uint64le><ann_record_1>...<ann_record_count><ann_id_1:uint64le>...<ann_id_count:uint64le>

    where ``count`` is the number of annotations that reference this related
    id, and each ``ann_record_k`` is the encoded geometry+property record
    of that annotation (per the spec, relationship lists are *omitted*
    from records in this index).

    We use pandas to subset → dropna → explode → sort the annotation columns,
    then encode the resulting frame in row order; each output segment's
    bytes are a contiguous slice into that single encoded buffer.
    """
    logger.info(f"Sorting annotations by relationship '{relationship}'")

    cols = _ann_required_cols(coord_space, annotation_type, property_specs) + [relationship]
    df = df[cols]
    if polyline_geom is not None:
        df['_polyline_pos'] = np.arange(len(df), dtype=np.int64)

    if not pd.api.types.is_integer_dtype(df[relationship]):
        df = df.dropna(subset=[relationship]).explode(relationship)

    # Sort by relationship so each segment's rows are contiguous. The spec
    # states that within a related-id group "the order of the annotations
    # does not matter", so a non-stable sort is fine here.
    df = df.sort_values(relationship)
    rel_values = df[relationship].to_numpy(np.uint64)

    # Permute polyline_geom into the exploded+sorted order,
    # then drop the helper column.
    if polyline_geom is not None:
        pos = df['_polyline_pos'].to_numpy()
        df = df.drop(columns='_polyline_pos')
        polyline_geom = PolylineGeometry(
            points=polyline_geom.points,
            starts=polyline_geom.starts[pos],
            ends=polyline_geom.ends[pos],
        )

    # Group boundaries: each unique relationship value becomes one output.
    if len(rel_values) == 0:
        boundaries = np.array([0], dtype=np.int64)
    else:
        boundaries = np.concatenate((
            [0],
            np.flatnonzero(rel_values[1:] != rel_values[:-1]) + 1,
            [len(rel_values)],
        ))
    unique_rels = rel_values[boundaries[:-1]]
    group_counts = (boundaries[1:] - boundaries[:-1]).astype(np.uint64)
    del rel_values

    # Encode the (now row-aligned) annotation records and id records.
    logger.info(f"Encoding annotations sorted by '{relationship}'")
    ann_pb = _encode_annotation_records(
        df, coord_space, annotation_type, property_specs, polyline_geom,
    )
    id_pb = _encode_id_bytes(df.index)
    del df

    # Translate per-row offsets to per-group offsets.
    if isinstance(ann_pb.layout, (int, np.integer)):
        group_ann_offsets = (boundaries * int(ann_pb.layout)).astype(np.int64)
    else:
        group_ann_offsets = ann_pb.layout[boundaries].astype(np.int64, copy=False)
    group_id_offsets = (boundaries * int(id_pb.layout)).astype(np.int64)

    buffers = [
        PartitionedBuffer(group_counts.astype('<u8', copy=False).tobytes(), 8),
        PartitionedBuffer(ann_pb.buf, group_ann_offsets),
        PartitionedBuffer(id_pb.buf, group_id_offsets),
    ]

    logger.info(f"Writing annotations to 'by_rel_{relationship}' index")
    metadata = _write_buffers(
        unique_rels, buffers,
        output_dir, f"by_rel_{relationship}",
        write_sharded, max_shards_per_transaction, max_threads,
    )
    metadata['id'] = relationship
    return metadata
