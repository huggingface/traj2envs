"""Publish a suite's environment images and pin them in each task.toml.

Training sandboxes start a task from a prebuilt image instead of building environment/, so,
like SmolDataEnvs, every task.toml gets

    [environment]
    docker_image = "docker.io/you/traj2envs@sha256:..."

For each task: build environment/ for the target platform, push it (tasks with identical
environments share one image), then run the oracle on that exact image (the healthcheck,
solution/solve.sh, then tests/test.sh) and pin the digest only if it earns full reward.

With `opencode_image` (built from assets/opencode.Dockerfile), the opencode binary is copied into
every image first, and into environment/Dockerfile so the Dockerfile still matches the image.
"""

from __future__ import annotations

import hashlib
import shlex
import tomllib
from pathlib import Path
from typing import Iterator

import tomli_w

from .docker import build_image, push_image, run_in_container

MAX_IMAGE_REFERENCE = 100  # HF sandboxes reject longer image references; a digest takes 72 of them


def oracle_reward(image: str, task_dir: Path, config: dict) -> str:
    """Reward the task's own solution earns in `image`, as the first line of reward.txt."""
    environment, verifier = config.get("environment", {}), config.get("verifier", {})
    exports = "".join(f"export {name}={shlex.quote(str(value))}\n" for name, value in verifier.get("env", {}).items())
    healthcheck = environment.get("healthcheck", {}).get("command", "true")
    script = (f"mkdir -p /workdir /logs/verifier\n{healthcheck}\n"
              f"bash /solution/solve.sh\n{exports}bash /tests/test.sh >/dev/null 2>&1\n"
              "echo; echo REWARD=$(head -n 1 /logs/verifier/reward.txt)")
    _, output = run_in_container(image, script, {task_dir / "tests": "/tests:ro", task_dir / "solution": "/solution:ro"},
                                 timeout_seconds=1800, network=bool(environment.get("allow_internet")))
    return next((line[len("REWARD="):] for line in reversed(output.splitlines()) if line.startswith("REWARD=")), "no reward")


def environment_hash(environment_dir: Path) -> str:
    """Hash of every file in environment/: tasks with the same hash get the same image."""
    digest = hashlib.sha256()
    for path in sorted(p for p in environment_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(environment_dir)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:12]


def publish_suite(suite_dir: Path, repository: str, verify: bool = True,
                  opencode_image: str | None = None) -> Iterator[dict]:
    """Push every task's image to `repository` and pin it; yield one record per task."""
    if len(repository) + len("@sha256:") + 64 > MAX_IMAGE_REFERENCE:
        raise ValueError(f"{repository}@sha256:<digest> would exceed {MAX_IMAGE_REFERENCE} characters: "
                         "use a shorter repository name")
    pinned_by_environment: dict[str, str] = {}
    for toml_path in sorted(suite_dir.glob("*/tasks/*/task.toml")):
        task_dir, family = toml_path.parent, toml_path.parent.parent.parent.name
        record = {"family": family, "task": task_dir.name, "pinned": False}
        config = tomllib.loads(toml_path.read_text())
        already_pinned = config.get("environment", {}).get("docker_image", "")
        if already_pinned.startswith(repository + "@"):  # resume: pinned by an earlier run
            yield {**record, "pinned": True, "reason": "already pinned", "docker_image": already_pinned}
            continue
        if opencode_image:
            dockerfile = task_dir / "environment" / "Dockerfile"
            copy_line = f"COPY --from={opencode_image} /usr/local/bin/opencode /usr/local/bin/opencode"
            if copy_line not in dockerfile.read_text():
                dockerfile.write_text(dockerfile.read_text().rstrip("\n") + f"\n{copy_line}\n")
        local_tag = f"traj2envs-publish-{family}-{task_dir.name}".lower()
        if error := build_image(local_tag, task_dir / "environment"):
            yield {**record, "reason": "image_build_failed", "detail": error}
            continue
        environment_id = environment_hash(task_dir / "environment")
        if environment_id not in pinned_by_environment:
            try:
                pinned_by_environment[environment_id] = push_image(local_tag, f"{repository}:{family}-{environment_id}")
            except RuntimeError as error:  # this task stays unpinned; re-running resumes it
                yield {**record, "reason": "push_failed", "detail": str(error)[-200:]}
                continue
        if verify and (reward := oracle_reward(local_tag, task_dir, config)) not in ("1", "1.0", "1.0000"):
            yield {**record, "reason": "oracle_failed_on_published_image", "detail": reward}
            continue
        config.setdefault("environment", {})["docker_image"] = pinned_by_environment[environment_id]
        toml_path.write_text(tomli_w.dumps(config))
        yield {**record, "pinned": True, "reason": "ok", "docker_image": pinned_by_environment[environment_id]}


def apply_pins(suite_dir: Path, log_text: str) -> tuple[int, list[str]]:
    """Write `docker_image` for every "PIN <family>/<task> <image>" line (from kaniko_publish.sh's log).

    Returns (tasks pinned, tasks the log reported as failed).
    """
    pinned, failed = 0, []
    for line in log_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "FAIL":
            failed.append(parts[1])
        if len(parts) != 3 or parts[0] != "PIN":
            continue
        family, task = parts[1].split("/", 1)
        toml_path = suite_dir / family / "tasks" / task / "task.toml"
        config = tomllib.loads(toml_path.read_text())
        config.setdefault("environment", {})["docker_image"] = parts[2]
        toml_path.write_text(tomli_w.dumps(config))
        pinned += 1
    return pinned, failed
