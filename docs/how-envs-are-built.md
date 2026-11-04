# How an environment is built

This page follows one `traj2envs run` from ATIF trajectories to a validated Harbor task, with the code
that does each step. Line numbers are as of this writing; function names are the stable reference.

```
cli.run ─► gate.scan_trajectory ─► Trajectory.load ─► task.make_tasks (per family) ─► suite.write_suite
                                                         │
          find_candidates → prepare → environment_files → build_image → check_in_image
                          → task_files → trials → run_trials → keep / Reject
```

## 1. Entry point: `cli.run`

[`traj2envs/cli.py:39`](../traj2envs/cli.py#L39)

- `find_trajectories` ([cli.py:26](../traj2envs/cli.py#L26)) collects every `.json` whose `schema_version`
  starts with `ATIF`.
- Trajectories are processed in parallel (`--jobs`, default 4) by `process` ([cli.py:49](../traj2envs/cli.py#L49)).
- Each trajectory first goes through `gate.scan_trajectory` ([gate.py:21](../traj2envs/gate.py#L21)): if
  `atif-scan` is installed, a trajectory with a blocking finding (reward tampering, benchmark leakage, …) is
  dropped with reason `gate`. `--no-scan` skips it.
- The trajectory is loaded with `Trajectory.load` and handed to `make_tasks` once per family in `--families`
  (registry: `FAMILIES` in [families/\_\_init\_\_.py](../traj2envs/families/__init__.py)). Tasks are written to
  `<out>/<family>/tasks/<task_id>/`.
- At the end, every record (kept or dropped) goes to `<out>/report.json` ([cli.py:64](../traj2envs/cli.py#L64))
  and `write_suite` lays out the dataset ([cli.py:65](../traj2envs/cli.py#L65)).

## 2. Replaying the trajectory

[`Trajectory.replay`, traj2envs/trajectory.py:171](../traj2envs/trajectory.py#L171)

Families never read raw steps to find code; they replay the trajectory. `replay(until_step=None)` walks every
tool call in order and keeps a `VirtualFilesystem` ([trajectory.py:61](../traj2envs/trajectory.py#L61)) up to date:

- File tools (`Write`, `Edit`, `MultiEdit`, full `Read`s, `apply_patch`) are applied by `apply_file_tool`
  ([trajectory.py:196](../traj2envs/trajectory.py#L196)). A full `Read` only seeds a file not seen before; an
  edit that can't be reconstructed marks the file as *lost*.
- Shell tools produce a `ShellRun` ([trajectory.py:37](../traj2envs/trajectory.py#L37)): step id, command,
  tracked working directory, parsed exit code, output, and a snapshot of the files **just before** the command
  ran (`files_before`). Afterwards, heredoc writes and inline patches in the command are applied by
  `apply_shell_file_changes` ([trajectory.py:217](../traj2envs/trajectory.py#L217)).

It returns `(shell_runs, final_filesystem)`. Other helpers families use: `user_messages`
([trajectory.py:157](../traj2envs/trajectory.py#L157)), `date_at` ([trajectory.py:167](../traj2envs/trajectory.py#L167)),
`short_id` ([trajectory.py:154](../traj2envs/trajectory.py#L154)).

## 3. The shared lifecycle: `make_tasks`

[`traj2envs/task.py:81`](../traj2envs/task.py#L81)

For each candidate returned by `family.find_candidates(trajectory)`:

| Step | Code | What happens |
|---|---|---|
| id + dedup | [task.py:84-90](../traj2envs/task.py#L84-L90) | `family.task_id(candidate)`; a task id already seen in this run → `Reject("duplicate")` |
| `prepare` | [task.py:91](../traj2envs/task.py#L91) | cheap checks and LLM calls, before any Docker work |
| environment | [task.py:92-93](../traj2envs/task.py#L92-L93) | `family.environment_files` written to `<task>/environment/` (at least a `Dockerfile`) |
| image build | [task.py:94-96](../traj2envs/task.py#L94-L96) | `docker.build_image` ([docker.py:27](../traj2envs/docker.py#L27)): `docker buildx build --load` for `linux/amd64` by default; failure → `Reject("image_build_failed")` |
| `check_in_image` | [task.py:97-98](../traj2envs/task.py#L97-L98) | checks that need the built image, using a scratch dir `<out>/<family>/tasks/.scratch/<id>/` |
| task files | [task.py:99](../traj2envs/task.py#L99) | `family.task_files` → `instruction.md`, `task.toml`, `tests/`, `solution/` |
| trials | [task.py:100-103](../traj2envs/task.py#L100-L103) | `family.trials` returns the trial matrix; `run_trials` runs it; any reward ≠ expected → `Reject("trials_failed", rewards)` |
| keep | [task.py:104](../traj2envs/task.py#L104) | yields `kept=True` plus `family.summary(candidate)` |
| drop | [task.py:105-109](../traj2envs/task.py#L105-L109) | any `Reject` deletes the task dir and yields `kept=False` with the reason; the scratch dir is always deleted |

### The trial runner

[`run_trials`, traj2envs/task.py:112](../traj2envs/task.py#L112)

Each `Trial(name, setup_script, expected_reward)` ([task.py:34](../traj2envs/task.py#L34)) runs in a fresh
container via `docker.run_in_container` ([docker.py:40](../traj2envs/docker.py#L40)) with **networking off**,
`tests/` mounted at `/tests:ro`, `solution/` at `/solution:ro`, plus any family-specific mounts. The script is:

```bash
<setup_script>
export <verifier env vars>      # family.verifier_env
bash /tests/test.sh
echo REWARD=$(head -n 1 /logs/verifier/reward.txt)
```

The reward is compared **as text** with `expected_reward`.

### `task.toml`

[`task_toml`, traj2envs/task.py:126](../traj2envs/task.py#L126) writes the Harbor config: `[task]` name
`traj2envs-<family>/<id>`, `[metadata]` (family + provenance such as `source_trace`), `[environment]`
(cpus, memory, `allow_internet`, optional `healthcheck`), `[verifier]` (timeout, env), `[agent]` timeout.

## 4. Family hooks

### repair — [traj2envs/families/repair.py](../traj2envs/families/repair.py)

| Hook | Line | Behaviour |
|---|---|---|
| `find_candidates` | [135](../traj2envs/families/repair.py#L135) | replays the whole trace; pairs a failing check (`is_fixable_failure`, [103](../traj2envs/families/repair.py#L103)) with the next run of the same check (`same_check`, [94](../traj2envs/families/repair.py#L94)) that exits 0. Repeated failures move the start to the latest one |
| `pair_to_candidate` | [155](../traj2envs/families/repair.py#L155) | drops fixes that needed an install, unknown/Windows roots, and fixes that only touched tests. Files at failure = failing run's `files_before`; files at success = passing run's `files_before`. Collects the successful setup commands before the failure |
| `task_id` | [171](../traj2envs/families/repair.py#L171) | `<short_id>-<sha256(fix)[:8]>` |
| `prepare` | [175](../traj2envs/families/repair.py#L175) | `Reject("fix_changes_check_config")` if the fix edits a check/config file |
| `environment_files` | [179](../traj2envs/families/repair.py#L179) | Dockerfile: `node:22-bookworm` or `python:3.12-bookworm`, `NPM_CONFIG_BEFORE=<trace date>`, scaffold commands, `COPY before/`, install commands, `COPY before/` again. `environment/before/` = files at failure |
| `check_in_image` | [200](../traj2envs/families/repair.py#L200) | the check must fail in the image (`check_passes_before_fix`) with overlapping error lines (`check_fails_differently`, via `error_lines` [121](../traj2envs/families/repair.py#L121)). Reads test/config files from the image to protect them |
| `task_files` | [213](../traj2envs/families/repair.py#L213) | `INSTRUCTION` ([54](../traj2envs/families/repair.py#L54)), `tests/test.sh` (`TEST_SH`, [43](../traj2envs/families/repair.py#L43): restore `tests/protected/`, run the check, 1 if exit 0), `solution/solve.sh` + `solution/files/` (the agent's fix) |
| `trials` | [240](../traj2envs/families/repair.py#L240) | `original_fix`→1, `original_fix_again`→1, `do_nothing`→0, `tamper_with_check` (`TAMPER_WITH_CHECK`, [50](../traj2envs/families/repair.py#L50))→0 |

### build — [traj2envs/families/build.py](../traj2envs/families/build.py)

| Hook | Line | Behaviour |
|---|---|---|
| `find_candidates` | [135](../traj2envs/families/build.py#L135) | replays up to the second user message; needs ≥2 `.py` files, no lost files, a last Python check that passed, no JS/TS/Go/Rust sources. Reference = files under the project root |
| `task_id` | [154](../traj2envs/families/build.py#L154) | `<short_id>-build` |
| `prepare` | [157](../traj2envs/families/build.py#L157) | `llm.ask` ([llm.py:17](../traj2envs/llm.py#L17)) with `SYSTEM_PROMPT` ([32](../traj2envs/families/build.py#L32)) returns `GeneratedTests` ([94](../traj2envs/families/build.py#L94)): contract, stub files, `test_hidden.py` |
| `environment_files` | [164](../traj2envs/families/build.py#L164) | `DOCKERFILE` ([50](../traj2envs/families/build.py#L50)): `python:3.12-bookworm` + pytest, empty `WORKDIR` |
| `check_in_image` | [167](../traj2envs/families/build.py#L167) | writes variants (reference ×2, stub, empty, mutants from `make_mutants`, [mutants.py:38](../traj2envs/mutants.py#L38)) to scratch and runs [assets/calibrate.py](../traj2envs/assets/calibrate.py) in the image. Keeps tests that pass on the reference twice and fail on stub and empty; requires ≥`MINIMUM_TESTS` and a mutant kill rate ≥`MINIMUM_MUTANT_KILL_RATE` ([27](../traj2envs/families/build.py#L27)), else `Reject("weak_tests")` |
| `task_files` | [199](../traj2envs/families/build.py#L199) | `INSTRUCTION` ([71](../traj2envs/families/build.py#L71)), `tests/test_hidden.py` filtered by `keep_only_tests` ([111](../traj2envs/families/build.py#L111)), `tests/test.sh` (`TEST_SH`, [55](../traj2envs/families/build.py#L55): fraction passed), `solution/` = the reference project |
| `trials` | [212](../traj2envs/families/build.py#L212) | `reference`→`1.0000`, `stub` (mounted at `/stub`, never shipped)→`0.0000`, `empty_workspace`→`0.0000` |

### qa — [traj2envs/families/qa.py](../traj2envs/families/qa.py)

| Hook | Line | Behaviour |
|---|---|---|
| `find_candidates` | [153](../traj2envs/families/qa.py#L153) | renders the trace (`render_for_llm`, [120](../traj2envs/families/qa.py#L120)) and asks the LLM (`SYSTEM_PROMPT`, [29](../traj2envs/families/qa.py#L29)) for 1-3 `ProposedQuestion`s ([94](../traj2envs/families/qa.py#L94)) |
| `task_id` | [157](../traj2envs/families/qa.py#L157) | `sha256(source + question)[:12]` |
| `prepare` | [160](../traj2envs/families/qa.py#L160) | the quote must be in the cited tool output (`cited_output`, [134](../traj2envs/families/qa.py#L134)) and contain the answer; the question must not contain it; every input is pinned to the commit current at the trace's first timestamp and downloaded (`pin_to_commit_at`, [140](../traj2envs/families/qa.py#L140)) |
| `environment_files` | [175](../traj2envs/families/qa.py#L175) | `DOCKERFILE` ([45](../traj2envs/families/qa.py#L45)), `pull_inputs.py` (`PULL_INPUTS`, [50](../traj2envs/families/qa.py#L50)), `inputs.json` (repo, path, commit, local name) |
| `image_tag` | [179](../traj2envs/families/qa.py#L179) | one shared local tag `traj2envs-qa`: inputs are downloaded by the healthcheck, not baked in |
| `verifier_env` | [182](../traj2envs/families/qa.py#L182) | `EXPECTED_ANSWER`, `QUESTION`, `REWARD_MODE`, `ATOL`, `RTOL` for the grader |
| `task_files` | [187](../traj2envs/families/qa.py#L187) | `INSTRUCTION` ([72](../traj2envs/families/qa.py#L72)), `tests/test.sh` (`TEST_SH`, [63](../traj2envs/families/qa.py#L63)) + `tests/grader.py` ([assets/grader.py](../traj2envs/assets/grader.py)), `solution/solve.py` + `solve.sh`. `task.toml` sets `allow_internet=true` and the healthcheck that runs `pull_inputs.py` |
| `trials` | [204](../traj2envs/families/qa.py#L204) | inputs are copied to scratch and mounted at `/home/user/input` (trials are offline); `reference_solution`→`1.0`, `wrong_answer`→`0.0`, `no_answer`→`0.0` |

## 5. Laying out the suite

[`write_suite`, traj2envs/suite.py:38](../traj2envs/suite.py#L38)

For each family with kept tasks: a Harbor dataset entry in `registry.json`, `<family>/manifest.parquet` (one row
per kept record), and a `configs:` subset in the dataset card `README.md`. The result is uploaded to the Hub
as a dataset (`hf upload … --repo-type dataset`).

## 6. After `run`: pinning images

Not part of building a task, but required before training sandboxes can start one.
`publish_suite` ([publish.py:48](../traj2envs/publish.py#L48)) rebuilds each `environment/`, pushes it to the
Docker registry given by `--repository` as `<repo>:<family>-<environment_hash>` (shared across identical
environments, [publish.py:40](../traj2envs/publish.py#L40)), re-runs the oracle on it (`oracle_reward`,
[publish.py:27](../traj2envs/publish.py#L27)), and writes `docker_image = "<repo>@sha256:…"` into `task.toml`.
[assets/kaniko_publish.sh](../traj2envs/assets/kaniko_publish.sh) + `apply_pins`
([publish.py:78](../traj2envs/publish.py#L78)) do the same from a hosted job, without the oracle re-check.
