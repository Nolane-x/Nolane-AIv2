from pathlib import Path
from nolane_ai.protocol.identity import source_tree_digest


def test_source_tree_digest_is_order_stable_and_content_sensitive(tmp_path: Path):
    (tmp_path / "src").mkdir(); (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "b.py").write_text("b=2\n", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("a=1\n", encoding="utf-8")
    first = source_tree_digest(tmp_path)
    second = source_tree_digest(tmp_path)
    assert first == second
    (tmp_path / "src" / "a.py").write_text("a=3\n", encoding="utf-8")
    assert source_tree_digest(tmp_path) != first
