# NGSidekick

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)][docs]

Tools for working with [neuroglancer][].  [See docs.][docs]

[neuroglancer]: https://github.com/google/neuroglancer
[docs]: https://janelia-flyem.github.io/ngsidekick/docs/index.html


## Feature Highlights

- **[Segment properties](https://janelia-flyem.github.io/ngsidekick/docs/segmentprops.html)**: read and write neuroglancer's
  precomputed [segment properties](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md)
  format from a pandas DataFrame.
- **[Local annotations](https://janelia-flyem.github.io/ngsidekick/docs/local.html)**: construct local annotations directly
  in viewer state. (API subject to change.)
- **[Precomputed annotations](https://janelia-flyem.github.io/ngsidekick/docs/precomputed.html)**: export annotations in
  neuroglancer's [precomputed annotations format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md)
  from a pandas DataFrame.

  - Supports all five annotation types:
    - `point`
    - `line`
    - `axis_aligned_bounding_box`
    - `ellipsoid`
    - `polyline`
  - Per-annotation **properties** (numeric, enum/categorical, and rgb/rgba color).
  - Per-annotation **relationships** (lists of related segment IDs, used by
    neuroglancer to filter annotations by segment).
  - Written to all "index" types (annotation id, related segment, and multi-level spatial grid)
  - Sharded output, written in parallel via [tensorstore](https://google.github.io/tensorstore/).
  - **Note:** writing directly to cloud storage is not yet supported;
    outputs must be written to a local filesystem (you must upload to cloud storage afterwards).

## Installation

Packages are available from both PyPI and conda-forge.

Using `pixi`:

```bash
pixi add ngsidekick
```

Using `conda`:

```bash
conda install -c conda-forge ngsidekick
```

Using pip:

```bash
pip install ngsidekick
```

Using uv:

```bash
uv add ngsidekick

# or in an existing environment
uv pip install ngsidekick
```
