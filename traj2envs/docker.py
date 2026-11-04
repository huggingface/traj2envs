"""The Docker operations the pipeline needs. Uses the caller's Docker context (set DOCKER_HOST if
the socket isn't the default).

Images are built for PLATFORM (linux/amd64 by default, what training sandboxes run) so the image
that passes the trials is the one that gets published. Set TRAJ2ENVS_PLATFORM=native to use the
host's architecture instead (faster on Apple silicon, but then validate before publishing).
"""

from __future__ import annotations

import io
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Callable

PLATFORM = os.environ.get("TRAJ2ENVS_PLATFORM", "linux/amd64")


def platform_args() -> list[str]:
    return [] if PLATFORM == "native" else ["--platform", PLATFORM]


def build_image(tag: str, context_dir: Path) -> str | None:
    """Build an image with BuildKit; return None on success, else the last line of the error.

    BuildKit (`docker buildx`, bundled with Docker Desktop) is needed to build for a platform other
    than the host's: the classic builder silently produces a host-platform image or fails.
    """
    process = subprocess.run(["docker", "buildx", "build", "--load", "-q", *platform_args(), "-t", tag, str(context_dir)],
                             capture_output=True, text=True)
    if process.returncode == 0:
        return None
    return (process.stderr.strip().splitlines() or ["docker build failed"])[-1][:300]


def run_in_container(image: str, script: str, mounts: dict[Path, str] | None = None,
                     timeout_seconds: int = 900, network: bool = False) -> tuple[int, str]:
    """Run a bash script in a fresh container (networking off unless asked); return (exit code, output)."""
    volume_args = []
    for host_path, container_path in (mounts or {}).items():
        volume_args += ["-v", f"{Path(host_path).resolve()}:{container_path}"]
    network_args = [] if network else ["--network", "none"]
    process = subprocess.run(["docker", "run", "--rm", *platform_args(), *network_args, *volume_args,
                              image, "bash", "-c", script], capture_output=True, text=True, timeout=timeout_seconds)
    return process.returncode, process.stdout + process.stderr


def read_files_from_image(image: str, directory: str, keep: Callable[[str], bool]) -> dict[str, str]:
    """Text files under `directory` in the image (skipping dependency folders) whose relative path passes `keep`."""
    list_and_archive = (f"cd {shlex.quote(directory)} && find . \\( -name node_modules -o -name .venv -o -name venv "
                        f"-o -name .git -o -name dist \\) -prune -o -type f -size -512k -print0 | tar --null -T - -cf -")
    process = subprocess.run(["docker", "run", "--rm", *platform_args(), "--network", "none", image, "bash", "-c",
                              list_and_archive], capture_output=True, timeout=300)
    files = {}
    if process.returncode != 0:
        return files
    with tarfile.open(fileobj=io.BytesIO(process.stdout)) as archive:
        for member in archive.getmembers():
            relative_path = member.name.removeprefix("./")
            if member.isfile() and keep(relative_path):
                try:
                    files[relative_path] = archive.extractfile(member).read().decode()
                except UnicodeDecodeError:
                    pass
    return files


def push_image(local_tag: str, remote_tag: str, timeout_seconds: int = 1200, attempts: int = 3) -> str:
    """Push under `remote_tag`; return the pinned reference `repository@sha256:...`.

    The digest is read from the push output, which every registry prints ("digest: sha256:..."):
    Docker's own records shorten names (docker.io/you/repo becomes you/repo), so they can't be matched.
    """
    subprocess.run(["docker", "tag", local_tag, remote_tag], check=True, capture_output=True)
    for attempt in range(1, attempts + 1):  # a push can hang on a stalled connection; retries resume it
        try:
            process = subprocess.run(["docker", "push", remote_tag], capture_output=True, text=True,
                                     timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process = subprocess.CompletedProcess([], returncode=1, stdout="", stderr=f"timed out after {timeout_seconds}s")
        if process.returncode == 0:
            break
        print(f"  push {remote_tag} attempt {attempt}/{attempts} failed: {process.stderr.strip()[-120:]}", flush=True)
    else:
        raise RuntimeError(f"docker push {remote_tag} failed: {process.stderr.strip()[-300:]}")
    pushed_digest = re.search(r"digest: (sha256:[0-9a-f]{64})", process.stdout + process.stderr)
    if pushed_digest is None:
        raise RuntimeError(f"docker push {remote_tag} printed no digest")
    repository = remote_tag.rsplit(":", 1)[0]
    return f"{repository}@{pushed_digest.group(1)}"
