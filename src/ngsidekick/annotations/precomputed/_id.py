import logging
from ._write_buffers import _write_buffers

logger = logging.getLogger(__name__)


def _write_annotations_by_id(df, output_dir, write_sharded, max_shards_per_transaction, max_threads):
    """
    Write the annotations to the "Annotation ID Index", a subdirectory of output_dir.

    Args:
        df:
            DataFrame with columns ['id_buf', 'ann_buf'] and optionally ['rel_buf'].

        output_dir:
            Directory to write the annotations to.
            A single subdirectory named 'by_id' will be created in output_dir.

        write_sharded:
            Whether to write the annotations in sharded format.

        max_shards_per_transaction, max_threads:
            See :func:`._write_buffers._write_buffers`.

    Returns:
        JSON metadata to be written under the 'by_id' key in the top-level 'info' file.
        Currently, this is always {"key": "by_id"}
    """
    if 'rel_buf' in df.columns:
        df = df[['ann_buf', 'rel_buf']]
    else:
        df = df[['ann_buf']]

    logger.info("Writing annotations to 'by_id' index")
    metadata = _write_buffers(
        df,
        output_dir,
        "by_id",
        write_sharded,
        max_shards_per_transaction,
        max_threads,
    )
    return metadata
