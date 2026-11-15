"""Run main.py on a GPU VM over SSH: upload the inputs, run there, download the facts.

The project must already be set up on the VM (deploy/verda/README.md, steps 2-3). Only the input files
travel on each run; they are streamed as tar archives through ssh, so no local tar or scp is needed.
"""

import logging
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("remote")

REMOTE_TARGET_ENV = "JUNCTION_REMOTE"  # e.g. root@1.2.3.4
REMOTE_DIR_ENV = (
    "JUNCTION_REMOTE_DIR"  # project folder on the VM, relative to the SSH home
)
DEFAULT_REMOTE_DIR = "junction"
SSH_KEY_ENV = "JUNCTION_SSH_KEY"  # path of the private key; unset = ssh's default (~/.ssh/id_ed25519, ...)
REMOTE_PYTHON = ".venv/bin/python"  # created by deploy/verda/setup.sh
RUN_DIR = "remote_run"  # inside the remote project folder, recreated every run
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",  # fail instead of asking for a password
    "-o",
    "StrictHostKeyChecking=accept-new",  # trust a new VM on first use, refuse a changed key
)
# user@host or host, host being a name, IPv4 or bracket-less IPv6; never starts with '-' (ssh option)
TARGET_PATTERN = re.compile(r"^(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9.:-]+$")
REMOTE_DIR_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class RemoteHost:
    target: str
    project_dir: str
    key_path: Path | None = None

    @property
    def input_dir(self) -> str:
        return f"{self.project_dir}/{RUN_DIR}/input"

    @property
    def output_dir(self) -> str:
        return f"{self.project_dir}/{RUN_DIR}/facts"


def remote_host(
    target: str, project_dir: str, key_path: str | None = None
) -> RemoteHost:
    """Validated host; all values end up on an ssh command line."""
    key = Path(key_path).expanduser() if key_path else None
    if key is not None and not key.is_file():
        raise ValueError(f"SSH key not found: {key} (check {SSH_KEY_ENV})")
    if not TARGET_PATTERN.match(target) or target.startswith("-"):
        raise ValueError(f"Invalid SSH target '{target}': expected user@host")
    if (
        not REMOTE_DIR_PATTERN.match(project_dir)
        or project_dir.startswith(("-", "/"))
        or ".." in project_dir
    ):
        raise ValueError(
            f"Invalid remote project folder '{project_dir}': use a path relative to the SSH home"
        )
    return RemoteHost(target, project_dir.rstrip("/"), key)


def ssh(host: RemoteHost, command: str, **kwargs) -> subprocess.Popen:
    key_option = ["-i", str(host.key_path)] if host.key_path else []
    return subprocess.Popen(
        ["ssh", *SSH_OPTIONS, *key_option, "--", host.target, command], **kwargs
    )


def wait_ok(proc: subprocess.Popen, step: str) -> None:
    if proc.wait() != 0:
        raise RuntimeError(f"Remote {step} failed (ssh exit code {proc.returncode})")


def upload(host: RemoteHost, files: list[Path]) -> None:
    """Replace the remote input folder with these files (flat, by file name)."""
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError("Two inputs share the same file name; put them in one folder")
    remote_dir = shlex.quote(host.input_dir)
    command = (
        f"rm -rf {remote_dir} && mkdir -p {remote_dir} && tar -xzf - -C {remote_dir}"
    )
    proc = ssh(host, command, stdin=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdin, mode="w|gz") as archive:
        for file in files:
            archive.add(file, arcname=file.name)
    proc.stdin.close()
    wait_ok(proc, "upload")


def run(host: RemoteHost, main_args: list[str]) -> None:
    """Run main.py on the VM on the uploaded inputs; its log streams to this terminal."""
    # main.py runs from the project folder, so it gets the run paths relative to it
    input_dir, facts_dir = f"{RUN_DIR}/input", f"{RUN_DIR}/facts"
    output_dir = shlex.quote(facts_dir)
    main_command = shlex.join(
        [REMOTE_PYTHON, "main.py", input_dir, "--output", facts_dir, *main_args]
    )
    command = (
        f"cd {shlex.quote(host.project_dir)} && rm -rf {output_dir} && mkdir -p {output_dir} "
        f"&& {main_command}"
    )
    wait_ok(ssh(host, command), "run")


def download(host: RemoteHost, local_dir: Path) -> None:
    """Copy the remote facts folder into local_dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    proc = ssh(
        host, f"tar -czf - -C {shlex.quote(host.output_dir)} .", stdout=subprocess.PIPE
    )
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as archive:
            # filter="data" refuses absolute paths and '..' coming from the VM
            archive.extractall(local_dir, filter="data")
    except tarfile.ReadError as error:
        proc.wait()
        raise RuntimeError(
            f"Remote download failed (ssh exit code {proc.returncode}): {error}"
        ) from error
    wait_ok(proc, "download")


def run_remote(
    host: RemoteHost, files: list[Path], local_output: Path, main_args: list[str]
) -> None:
    logger.info(
        "Uploading %d file(s) to %s:%s", len(files), host.target, host.input_dir
    )
    upload(host, files)
    logger.info("Running main.py on %s", host.target)
    run(host, main_args)
    download(host, local_output)
    logger.info("Facts copied to %s", local_output)
