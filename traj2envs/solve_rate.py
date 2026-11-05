"""Solve rates: how often a reference agent solves each task, measured with Harbor.

A task every attempt solves teaches little; one no attempt solves may be out of reach (or
broken). `measure` runs `harbor run` once per family with k attempts per task, and records each
task's solve rate (mean reward over the attempts that reached the verifier) in its task.toml
and the family manifest, so the suite can be filtered by difficulty. Attempts that crashed
before grading (infrastructure, API errors) are counted as errors, not failures.

A Hub model id (`Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest`) is served by Inference Providers
through the OpenAI-compatible router, with HF_TOKEN; any other id is passed to Harbor as is.
"""

from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w

HF_ROUTER_URL = "https://router.huggingface.co/v1"
HARBOR_VERSION = "0.25.0"


def harbor_command() -> list[str]:
    return ["harbor"] if shutil.which("harbor") else ["uvx", "--from", f"harbor=={HARBOR_VERSION}", "harbor"]


def model_args(model: str | None, base_url: str | None = None) -> list[str]:
    """A Hub id goes to the HF router, or to `base_url` (any OpenAI-compatible server: vLLM, an Inference Endpoint)."""
    if not model:
        return []
    if base_url or ("/" in model and model.split("/", 1)[0] not in ("anthropic", "gemini", "openrouter")):  # openai/ is a Hub org too
        return ["-m", f"openai/{model}", "--base-url", base_url or HF_ROUTER_URL, "--api-key-name", "HF_TOKEN",
                "--api-format", "openai_chat_completions"]
    return ["-m", model]


def run_harbor(tasks_dir: Path, agent: str, model: str | None, attempts: int, concurrency: int,
               jobs_dir: Path, job_name: str, base_url: str | None = None, environment: str = "docker",
               agent_kwargs: list[str] = ()) -> Path:
    # terminus-2 records its terminal with asciinema, which it would install at start: tasks are offline
    agent_args = ["--ak", "record_terminal_session=false"] if agent == "terminus-2" else []
    agent_args += [arg for kwarg in agent_kwargs for arg in ("--ak", kwarg)]
    subprocess.run([*harbor_command(), "run", "-p", str(tasks_dir), "-a", agent, *agent_args, *model_args(model, base_url),
                    "--env", environment,
                    "-k", str(attempts), "-n", str(concurrency), "-o", str(jobs_dir), "--job-name", job_name,
                    "-y", "-q"], check=True)
    return jobs_dir / job_name


def read_rewards(job_dir: Path) -> dict[str, list[float | None]]:
    """{task_id: [reward, or None for an attempt that never reached the verifier]}."""
    rewards = collections.defaultdict(list)
    for result_path in sorted(job_dir.glob("*/result.json")):
        trial = json.loads(result_path.read_text())
        task_id = Path(trial["task_id"]["path"]).name
        reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rewards[task_id].append(None if reward is None else float(reward))
    return rewards


def record(suite: Path, family: str, rewards: dict[str, list[float | None]], agent: str, model: str | None) -> list[dict]:
    """Write solve rates into each task.toml and the family manifest; return one row per task."""
    rows = []
    for task_id, task_rewards in sorted(rewards.items()):
        scored = [reward for reward in task_rewards if reward is not None]
        row = {"family": family, "task_id": task_id, "rewards": task_rewards, "errors": len(task_rewards) - len(scored),
               "solve_rate": round(sum(scored) / len(scored), 4) if scored else None}
        rows.append(row)
        task_toml = suite / family / "tasks" / task_id / "task.toml"
        config = tomllib.loads(task_toml.read_text())
        config["metadata"].update({"solve_rate_agent": f"{agent}:{model}" if model else agent,
                                   "solve_rate_attempts": len(scored)})
        if row["solve_rate"] is not None:
            config["metadata"]["solve_rate"] = row["solve_rate"]
        task_toml.write_text(tomli_w.dumps(config))

    manifest_path = suite / family / "manifest.parquet"
    if manifest_path.exists():
        by_task = {row["task_id"]: row for row in rows}
        manifest = pq.read_table(manifest_path).to_pylist()
        for entry in manifest:
            row = by_task.get(entry["task_id"])
            entry["solve_rate"] = str(row["solve_rate"]) if row else entry.get("solve_rate", "None")
        pq.write_table(pa.Table.from_pylist(manifest), manifest_path)
    return rows


def measure(suite: Path, families: list[str], agent: str, model: str | None, attempts: int, concurrency: int,
            jobs_dir: Path, base_url: str | None = None, environment: str = "docker",
            agent_kwargs: list[str] = ()) -> list[dict]:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{agent}-{model or ''}").strip("-")[:60]
    rows = []
    for family in families:
        tasks_dir = suite / family / "tasks"
        if not any(tasks_dir.glob("*/task.toml")):  # Harbor exits with an error on a folder with no tasks
            continue
        job_dir = run_harbor(tasks_dir, agent, model, attempts, concurrency, jobs_dir,
                             f"solve-rate-{family}-{slug}-{time.strftime('%Y%m%d-%H%M%S')}", base_url, environment,
                             agent_kwargs)
        rows += record(suite, family, read_rewards(job_dir), agent, model)
    (suite / "solve_rates.json").write_text(json.dumps({"agent": agent, "model": model, "attempts": attempts,
                                                        "tasks": rows}, indent=1))
    return rows
