from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tree_digest(root: str | Path) -> str:
    root = Path(root)
    files: list[Path] = []
    for relative_root in ("src", "scripts"):
        base = root / relative_root
        if base.exists():
            files.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
