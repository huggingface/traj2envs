# traj2envs

Turn [ATIF](https://github.com/laude-institute/harbor) agent trajectories into validated
[Harbor](https://github.com/laude-institute/harbor) RL tasks.

```bash
pip install -e ".[llm]"        # the llm extra is needed by the qa and build families
traj2envs run trajectories/ --families repair,build,qa --out suite/
traj2envs publish suite/ --repository docker.io/<you>/traj2envs   # after `docker login`
hf upload <you>/my-envs suite/ . --repo-type dataset --private
```

`publish` pushes each task's environment image and pins it in `task.toml`
(`docker_image = "...@sha256:..."`), which training sandboxes need to start a task without
building it. Tasks with identical environments share one image, and each pinned image is
verified first by running the task's oracle in it. Images are built for `linux/amd64` by
default (`--platform` to change it). Images in a public registry expose the task contents
even if the dataset repo is private.

On a slow uplink or an Apple-silicon laptop, push from a hosted job instead. `assets/kaniko_publish.sh`
builds and pushes every environment with Kaniko (no Docker daemon needed), skips tasks already
pinned, and prints a `PIN` line per task; `traj2envs pin` applies them:

```bash
export DOCKER_USERNAME=<you> DOCKER_PASSWORD=<registry access token>
hf jobs run -d --flavor cpu-upgrade --timeout 6h -v ./suite:/suite -v ./traj2envs/assets:/scripts \
  --secrets DOCKER_USERNAME --secrets DOCKER_PASSWORD gcr.io/kaniko-project/executor:v1.23.2-debug \
  /busybox/sh /scripts/kaniko_publish.sh /suite docker.io/<you>/traj2envs
hf jobs logs <job-id> > publish.log && traj2envs pin suite/ publish.log
```

This path skips the oracle re-check that `publish` runs, so validate on `linux/amd64` first
(`traj2envs run` does by default).

HF Jobs can't run containers (no Docker daemon, no privileged mode), so `run` itself needs a
machine with Docker. The LLM calls can still go to a job: `--propose-only` stops every candidate
before Docker, filling the LLM cache, and a later `run` with the same `TRAJ2ENVS_MODEL` replays
the cache instead of calling the model:

```bash
hf buckets create <you>/traj2envs-cache --private
hf jobs run -d --flavor cpu-upgrade --timeout 6h \
  -v ./traj2envs:/src/traj2envs -v ./trajectories:/trajectories -v hf://buckets/<you>/traj2envs-cache:/cache \
  --secrets HF_TOKEN -e TRAJ2ENVS_MODEL=<org/model:provider> -e TRAJ2ENVS_CACHE=/cache -e PYTHONPATH=/src \
  python:3.12-slim sh -c 'pip install -q huggingface_hub pyarrow pydantic tomli-w && \
    python -m traj2envs.cli run /trajectories --families build,qa --propose-only --no-scan --jobs 16 --out /tmp/proposals'
hf buckets sync hf://buckets/<you>/traj2envs-cache ~/.cache/traj2envs/llm
TRAJ2ENVS_MODEL=<org/model:provider> traj2envs run trajectories/ --families repair,build,qa --out suite/
```

Every call's tokens and cost are recorded in `$TRAJ2ENVS_CACHE/usage/`. `TRAJ2ENVS_BUDGET_USD=50`
stops new calls once that total reaches $50 (shared by every job using the same cache;
`TRAJ2ENVS_PRICES=0.70,2.10` prices tokens for providers that don't report costs). Reasoning
models spend output tokens thinking: `TRAJ2ENVS_REASONING_EFFORT=low` and `TRAJ2ENVS_MAX_TOKENS`
(default 32000) keep them from running out before they answer.

`suite/` is a Hub dataset with one subset per family, a Harbor `registry.json`, and a
`report.json` that says why each candidate was kept or dropped. Add `--limit N` to process only
the first N trajectories while debugging.

## Difficulty

`solve-rate` runs a reference agent on every task with [Harbor](https://github.com/laude-institute/harbor)
(k attempts each, in the task's own container) and records the mean reward as `solve_rate` in
`task.toml`, the manifests and `solve_rates.json`. Tasks every attempt solves teach little; aim
for a band such as 0.1-0.7. A Hub model id is served by Inference Providers (needs `HF_TOKEN`):

```bash
traj2envs solve-rate suite/ --model Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest -k 4
traj2envs solve-rate suite/ --agent oracle -k 1     # sanity check: every task should score 1
```

The default agent is Harbor's `terminus-2`; any Harbor agent works (`--agent claude-code`, ...).
Attempts that crash before grading count as errors, not failures. Harbor is run with `uvx` if
it isn't installed.

## Families

| Family | Found in a trajectory | The agent must | Reward |
|---|---|---|---|
| `repair` | a check that failed, then passed after file edits | make the check pass, starting from the files at the failure | 1 or 0 |
| `build` | a Python project finished within the first request | build it from the request and an interface contract | fraction of hidden tests passed |
| `extend` | a later request that changed a Python project and ended with its checks passing | implement that request in the project as it was before it | fraction of new hidden tests passed; 0 if existing behaviour breaks |
| `qa` | a fact the agent computed from public Hugging Face data | answer the question from the staged files | exact or numeric match |

## How it works

`trajectory.py` replays a trajectory: file changes (`Write`, `Edit`, `apply_patch`, heredocs,
full `Read`s) are applied to a virtual filesystem, so every shell command comes with the files
as they were just before it ran.

`task.py` runs every candidate through the same lifecycle:

```
find_candidates → prepare → build environment image → check_in_image → write task files → trials
```

A family fills in those hooks and raises `Reject("reason")` to drop a candidate. A task is kept
only if its **trial matrix** passes in its own Docker image, offline:

| Family | Must score full reward | Must score 0 |
|---|---|---|
| `repair` | the original fix (twice) | doing nothing; rewriting the check's tests or config |
| `build` | the reference implementation | an empty workspace; a stub with no behaviour |
| `extend` | the original implementation (twice) | doing nothing; deleting the project |
| `qa` | the reference solution | a wrong answer; no answer |

Before that, each family checks what it can: `repair` requires the rebuilt project to fail the way
the trajectory recorded; `build` keeps only tests that pass on the reference and fail on the stub
and the empty workspace, and requires them to catch most injected bugs (`mutants.py`); `qa`
requires the answer to appear verbatim in the cited tool output and pins every input file to the
dataset commit at the trajectory's date.

The LLM (`claude-opus-5-5`, override with `TRAJ2ENVS_MODEL`) only proposes questions, contracts
and tests. A Hub model id (`TRAJ2ENVS_MODEL=moonshotai/Kimi-K2-Instruct:groq`, or `:fastest` /
`:cheapest`) goes through [Inference Providers](https://huggingface.co/docs/inference-providers)
with `HF_TOKEN` (`TRAJ2ENVS_BILL_TO=<org>` to bill an org). Responses are cached in
`~/.cache/traj2envs/llm`, so re-runs are free.

[docs/how-envs-are-built.md](docs/how-envs-are-built.md) walks through the whole pipeline with
the code behind each step.

## Adding a family

```python
class Lint(Family):
    name = "lint"

    def find_candidates(self, trajectory): ...      # -> list of your candidate objects
    def task_id(self, candidate): ...
    def environment_files(self, candidate): ...     # -> {"Dockerfile": ..., ...}
    def task_files(self, candidate): ...            # -> instruction.md, task.toml, tests/, solution/
    def trials(self, candidate, scratch_dir): ...   # -> [Trial(name, setup_script, expected_reward)], mounts
```

Optional hooks: `prepare` (cheap checks before Docker), `check_in_image`, `verifier_env`,
`summary`. Register it in `families/__init__.py`.

## Raw session logs

```bash
traj2envs convert ~/.claude/projects/*/*.jsonl --format claude-code --out trajectories/
traj2envs convert rollout-*.jsonl --format codex --out trajectories/
traj2envs convert hf://datasets/TeichAI/DeepSeek-v4-Pro-Agent --format pi --limit 500 --out trajectories/
```

An `hf://datasets/<repo>[/<glob>]` input downloads the dataset's session files (`*.jsonl`) from the Hub.
Trace datasets mostly hold one-shot "build X" sessions: expect a few percent of sessions to yield a
`repair` or `build` candidate.

## Notes

- Needs Docker (set `DOCKER_HOST` if your socket isn't the default); `qa` and `build` also need
  `ANTHROPIC_API_KEY` (or `HF_TOKEN` for a Hub model). [atif-scan](https://github.com/huggingface/atif-scan), if installed, screens
  trajectories first (`--no-scan` to skip).
- Tasks from the same trajectory can be near-duplicates: split train/eval by `source_trace`.
- Trace datasets often carry no license; publish privately until redistribution is cleared.
- `assets/grader.py` is the deterministic grader from
  [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train) (MIT).
