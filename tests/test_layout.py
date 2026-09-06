from importlib.util import find_spec


def test_research_packages_exist():
    assert find_spec("nolane_ai.model.budget") is not None
    assert find_spec("nolane_ai.protocol.schema") is not None
