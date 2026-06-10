"""
neuroglancer-related utility functions
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ngsidekick")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

from .gcs import upload_ngstate, upload_ngstates, upload_json, upload_to_bucket, make_bucket_public
from .state_utils import parse_nglink, encode_ngstate, layer_dict, layer_state, download_ngstate
from .annotations.local import local_annotation_json, extract_local_annotations
from .annotations.precomputed import write_precomputed_annotations
from .segmentprops import segment_properties_json, segment_properties_to_dataframe
from .segmentcolors import hex_string_from_segment_id
from .cors_server import serve_directory
