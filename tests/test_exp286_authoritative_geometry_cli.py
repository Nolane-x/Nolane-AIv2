from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_exp286_paired_dev.py"
MANIFEST = ROOT / "protocols" / "exp286_development_geometry_v1.json"
DIGEST = ROOT / "protocols" / "exp286_development_geometry_v1.sha256"


def _outputs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "execution.json", tmp_path / "registry.json"


def test_exp286_cli_rejects_tiny_with_authoritative_geometry(tmp_path: Path) -> None:
    output, registry = _outputs(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--geometry-manifest", str(MANIFEST),
            "--geometry-digest-file", str(DIGEST),
            "--tiny",
            "--output", str(output),
            "--registry-output", str(registry),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "geometry manifest cannot be combined with --tiny" in completed.stderr
    assert not output.exists()
    assert not registry.exists()


def test_exp286_cli_verifies_geometry_digest_before_execution(tmp_path: Path) -> None:
    output, registry = _outputs(tmp_path)
    manifest = tmp_path / "geometry.json"
    manifest.write_bytes(MANIFEST.read_bytes() + b"\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--geometry-manifest", str(manifest),
            "--geometry-digest-file", str(DIGEST),
            "--output", str(output),
            "--registry-output", str(registry),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "geometry digest mismatch" in completed.stderr
    assert not output.exists()
    assert not registry.exists()
