# NGSidekick

Tools for neuroglancer scenes.


## Installation

- Using uv:

```bash
uv add ngsidekick
# or in an existing environment
uv pip install ngsidekick
```

- Using conda (conda-forge):

```bash
conda install -c conda-forge ngsidekick
```

- Using pixi (conda-forge):

```bash
pixi add ngsidekick -c conda-forge
```

## Development

Create an environment and run tests:

```bash
uv venv
uv pip install -e .[test]
pytest
```

## License

BSD-3-Clause; see `LICENSE`.

