import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_simple_datagen_import_path_has_curobo_available() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    simple_src = repo_root / "third_party" / "SIMPLE" / "src"
    assert simple_src.is_dir(), "third_party/SIMPLE submodule must be initialized"

    script = textwrap.dedent(
        """
        from curobo.types.base import TensorDeviceType
        import simple.envs

        assert TensorDeviceType is not None
        assert simple.envs is not None
        """
    )

    env = os.environ.copy()
    pythonpath = str(simple_src)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
