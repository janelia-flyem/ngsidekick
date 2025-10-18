import ngsidekick


def test_version_is_defined() -> None:
    assert isinstance(ngsidekick.__version__, str)
    assert len(ngsidekick.__version__) > 0


