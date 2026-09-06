from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_verifier_runs_from_fresh_checkout_without_install():
    result = subprocess.run(
        [sys.executable, "scripts/verify_protocol.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "NLM-REASONING-STAGE-A-CONFIRMATORY-V1" in result.stdout
