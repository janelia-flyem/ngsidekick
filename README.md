# NGSidekick

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)][docs]

Tools for neuroglancer scenes.  [See docs.][docs]

[docs]: https://janelia-flyem.github.io/ngsidekick/docs/index.html


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

## Development

Create an environment and run tests:

```bash
uv venv
uv pip install -e .[test]
pytest
```
