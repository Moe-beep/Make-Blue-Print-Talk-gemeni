"""Offline test of main.py --remote: ssh replaced by a local bash, the VM's main.py by a stub.

Run: python tests/test_remote_offline.py   (needs bash and tar on PATH, e.g. Git Bash on Windows)
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines import remote

# stands in for .venv/bin/python on the VM: "python main.py <input> --output <dir> ..." -> one file per input
FAKE_PYTHON = """#!/usr/bin/env bash
input="$2"; output="$4"
for f in "$input"/*; do echo "{\\"image\\": \\"$(basename "$f")\\", \\"args\\": \\"$*\\"}" > "$output/$(basename "${f%.*}").json"; done
"""


def test_validation() -> None:
    assert remote.remote_host("root@1.2.3.4", "junction").target == "root@1.2.3.4"
    assert remote.remote_host("my-vm", "work/junction/").project_dir == "work/junction"
    for target in ("-oProxyCommand=x", "root@host; rm -rf /", "", "a b"):
        try:
            remote.remote_host(target, "junction")
            raise AssertionError(f"accepted target {target!r}")
        except ValueError:
            pass
    key = Path(__file__)  # any existing file stands in for a key
    assert remote.remote_host("root@h", "junction", str(key)).key_path == key
    try:
        remote.remote_host("root@h", "junction", "missing/id_ed25519")
        raise AssertionError("accepted a missing key")
    except ValueError:
        pass
    for folder in ("/root/junction", "../x", "a;b", "-x"):
        try:
            remote.remote_host("root@h", folder)
            raise AssertionError(f"accepted folder {folder!r}")
        except ValueError:
            pass


def test_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        home, local = tmp / "home", tmp / "local"
        venv_bin = home / "junction" / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text(FAKE_PYTHON, newline="\n")
        (venv_bin / "python").chmod(0o755)
        local.mkdir()
        plans = [local / "plan a.png", local / "b.pdf"]
        for plan in plans:
            plan.write_bytes(b"x")

        bash = shutil.which(
            "bash"
        )  # PATH order: on Windows, Git Bash's, not System32's WSL launcher
        assert bash, "bash not found on PATH"

        def fake_ssh(host, command, **kwargs):
            return subprocess.Popen([bash, "-c", command], cwd=home, **kwargs)

        remote.ssh = fake_ssh
        host = remote.remote_host("root@1.2.3.4", "junction")
        out = local / "facts"
        remote.run_remote(host, plans, out, ["--device", "cuda", "--models", "mitunet"])

        written = sorted(p.name for p in out.glob("*.json"))
        assert written == ["b.json", "plan a.json"], written
        assert "--device cuda --models mitunet" in (out / "b.json").read_text()


if __name__ == "__main__":
    test_validation()
    test_round_trip()
    print("ok")
