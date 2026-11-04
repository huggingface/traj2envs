"""traj2envs: turn ATIF agent trajectories into validated Harbor RL tasks.

  traj2envs run TRAJECTORIES... --families repair,build,qa --out suite/
  traj2envs run TRAJECTORIES... --families build,qa --out proposals/ --propose-only   (LLM calls only, no Docker)
  traj2envs publish suite/ --repository docker.io/you/traj2envs
  traj2envs pin suite/ job.log          (pins pushed by assets/kaniko_publish.sh, e.g. from HF Jobs)
  traj2envs solve-rate suite/ --model Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest -k 4   (difficulty, via Harbor)
  traj2envs convert SESSION.jsonl... --format claude-code|codex|pi --out trajectories/
  traj2envs convert hf://datasets/ORG/NAME[/GLOB] --format pi --limit 50 --out trajectories/
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
from pathlib import Path

from . import docker
from .families import FAMILIES
from .gate import scan_trajectory
from .suite import write_suite
from .task import make_tasks
from .trajectory import Trajectory


def find_trajectories(inputs: list[str]) -> list[Path]:
    """ATIF .json files among the given files and directories (searched recursively)."""
    found = []
    for item in map(Path, inputs):
        for path in sorted(item.rglob("*.json")) if item.is_dir() else [item]:
            try:
                if str(json.loads(path.read_text()).get("schema_version", "")).startswith("ATIF"):
                    found.append(path)
            except (ValueError, UnicodeDecodeError, AttributeError):
                pass
    return found


def run(args: argparse.Namespace) -> None:
    out_dir, family_names = Path(args.out), args.families.split(",")
    if unknown := set(family_names) - set(FAMILIES):
        sys.exit(f"unknown families {sorted(unknown)}; available: {sorted(FAMILIES)}")
    trajectory_paths = find_trajectories(args.inputs)
    if args.limit is not None:
        trajectory_paths = trajectory_paths[:args.limit]
    from .llm import MODEL
    print(f"{len(trajectory_paths)} trajectories, families: {', '.join(family_names)}, LLM: {MODEL}")
    seen_task_ids = {name: set() for name in family_names}

    def process(path: Path) -> list[dict]:
        passed_gate, gate_result = (True, "skipped (--no-scan)") if args.no_scan else scan_trajectory(path)
        if not passed_gate:
            return [{"trajectory": str(path), "family": "-", "kept": False, "reason": "gate", "detail": gate_result}]
        trajectory, records = Trajectory.load(path), []
        for name in family_names:
            for record in make_tasks(FAMILIES[name], trajectory, out_dir / name / "tasks", seen_task_ids[name],
                                     args.propose_only):
                records.append({"trajectory": str(path), "gate": gate_result, **record})
                status = "kept" if record["kept"] else "prop" if record["reason"] == "proposed" else "drop"
                detail = f"  ({record['detail']})" if record["reason"] in ("error", "budget_exhausted") else ""
                print(f"{status}  {name:<7} {record['reason']:<28} {path.name}{detail}", flush=True)
        return records

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        records = [record for batch in pool.map(process, trajectory_paths) for record in batch]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(records, indent=1, default=str))
    if args.propose_only:
        print(f"\n{sum(r['reason'] == 'proposed' for r in records)} candidates proposed; LLM responses are in the cache")
        return
    write_suite(out_dir, {name: [r for r in records if r["family"] == name and r["kept"]] for name in family_names})
    print("\nfunnel:")
    for (family, reason), count in sorted(collections.Counter((r["family"], r["reason"]) for r in records).items()):
        print(f"  {family:<7} {reason:<28} {count}")
    print(f"\n{sum(r['kept'] for r in records)} tasks in {out_dir} (report.json explains every candidate)")


def publish(args: argparse.Namespace) -> None:
    from .publish import publish_suite

    records = []
    for record in publish_suite(Path(args.suite), args.repository, verify=not args.no_verify):
        records.append(record)
        print(f"{'pinned' if record['pinned'] else 'FAILED'}  {record['family']:<7} {record['task']:<40} "
              f"{record.get('docker_image') or record['reason'] + ' ' + str(record.get('detail', ''))}", flush=True)
    print(f"\n{sum(r['pinned'] for r in records)}/{len(records)} tasks pinned to an image in {args.repository}")


def pin(args: argparse.Namespace) -> None:
    from .publish import apply_pins

    pinned, failed = apply_pins(Path(args.suite), Path(args.log).read_text())
    print(f"pinned {pinned} tasks" + (f"; failed in the job: {', '.join(failed)}" if failed else ""))


def find_sessions(inputs: list[str], limit: int | None) -> list[tuple[str, str]]:
    """(local path, source reference) per session; hf://datasets/<repo>[/<glob>] is downloaded from the Hub."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    sessions = []
    for item in inputs:
        if not item.startswith("hf://"):
            sessions.append((item, str(Path(item).resolve())))
            continue
        filesystem = HfFileSystem()
        pattern = item if any(char in item for char in "*?[") else item.rstrip("/") + "/**/*.jsonl"
        for remote_path in sorted(filesystem.glob(pattern))[:None if limit is None else max(0, limit - len(sessions))]:
            resolved = filesystem.resolve_path(remote_path)
            local_path = hf_hub_download(resolved.repo_id, resolved.path_in_repo, repo_type=resolved.repo_type,
                                         revision=resolved.revision)
            sessions.append((local_path, f"hf://{remote_path}"))
    return sessions[:limit]


def solve_rate(args: argparse.Namespace) -> None:
    from .solve_rate import measure

    suite = Path(args.suite)
    if not suite.is_dir():
        sys.exit(f"{suite} does not exist: generate it first with `traj2envs run ... --out {suite}`")
    families = args.families.split(",") if args.families else sorted(
        path.name for path in suite.iterdir() if (path / "tasks").is_dir())
    rows = measure(suite, families, args.agent, args.model, args.attempts, args.concurrency, Path(args.jobs_dir),
                   args.base_url, args.env)
    low, high = map(float, args.band.split(","))
    print(f"\n{'task':<42} {'family':<7} {'solve rate':>10}  rewards")
    for row in sorted(rows, key=lambda row: (row["solve_rate"] is None, row["solve_rate"] or 0)):
        rate = row["solve_rate"]
        verdict = ("errored" if rate is None else "too easy" if rate > high else "unsolved" if rate == 0
                   else "too hard" if rate < low else "in band")
        rewards = " ".join("err" if reward is None else f"{reward:g}" for reward in row["rewards"])
        print(f"{row['task_id']:<42} {row['family']:<7} {'-' if rate is None else f'{rate:.2f}':>10}  {rewards}  ({verdict})")
    in_band = sum(row["solve_rate"] is not None and low <= row["solve_rate"] <= high for row in rows)
    print(f"\n{in_band}/{len(rows)} tasks in the {low:g}-{high:g} band; solve rates written to task.toml, "
          f"the manifests and {suite / 'solve_rates.json'}")


def convert(args: argparse.Namespace) -> None:
    from .adapters import convert as convert_session

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for session, source_ref in find_sessions(args.sessions, args.limit):
        trajectory = convert_session(session, args.format, source_ref)
        destination = out_dir / f"{Path(session).stem}.json"
        destination.write_text(json.dumps(trajectory, indent=1, ensure_ascii=False))
        print(f"{session} -> {destination} ({len(trajectory['steps'])} steps)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="traj2envs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)

    run_parser = commands.add_parser("run", help="ATIF trajectories -> validated Harbor tasks")
    run_parser.add_argument("inputs", nargs="+", help="ATIF .json files or directories")
    run_parser.add_argument("--families", default="repair,build,qa")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--jobs", type=int, default=4, help="trajectories processed in parallel")
    run_parser.add_argument("--limit", type=int, help="only process the first N trajectories (for debugging)")
    run_parser.add_argument("--no-scan", action="store_true", help="skip the atif-scan gate")
    run_parser.add_argument("--propose-only", action="store_true",
                            help="only make the LLM calls (filling TRAJ2ENVS_CACHE), no Docker: e.g. in an HF Job")
    run_parser.add_argument("--platform", default=docker.PLATFORM, help="image platform, or 'native' (default: %(default)s)")
    run_parser.set_defaults(handler=run)

    publish_parser = commands.add_parser("publish", help="push each task's image and pin it in task.toml (docker_image)")
    publish_parser.add_argument("suite", help="a suite folder written by `run`")
    publish_parser.add_argument("--repository", required=True, help="e.g. docker.io/you/traj2envs (run `docker login` first)")
    publish_parser.add_argument("--platform", default=docker.PLATFORM, help="image platform, or 'native' (default: %(default)s)")
    publish_parser.add_argument("--no-verify", action="store_true", help="skip re-running the oracle on the published image")
    publish_parser.set_defaults(handler=publish)

    pin_parser = commands.add_parser("pin", help="write docker_image from a kaniko_publish.sh log (e.g. an HF Jobs log)")
    pin_parser.add_argument("suite")
    pin_parser.add_argument("log", help="file with the job's output (PIN lines)")
    pin_parser.set_defaults(handler=pin)

    rate_parser = commands.add_parser("solve-rate", help="run a reference agent k times per task (Harbor); record solve rates")
    rate_parser.add_argument("suite", help="a suite folder written by `run`")
    rate_parser.add_argument("--model", help="Hub id (served by Inference Providers, needs HF_TOKEN) or a Harbor/LiteLLM model")
    rate_parser.add_argument("--base-url", help="serve --model from this OpenAI-compatible URL instead (vLLM, an Inference "
                             "Endpoint, ...); HF_TOKEN is sent as the key")
    rate_parser.add_argument("--env", default="docker", help="where Harbor runs each attempt: docker (local) or a remote "
                             "sandbox (modal, daytona, e2b, ...; needs that provider's API key) (default: %(default)s)")
    rate_parser.add_argument("--agent", default="terminus-2", help="any Harbor agent (default: %(default)s; `oracle` checks the tasks)")
    rate_parser.add_argument("-k", "--attempts", type=int, default=4, help="attempts per task (default: %(default)s)")
    rate_parser.add_argument("-n", "--concurrency", type=int, default=4, help="trials run at once (default: %(default)s)")
    rate_parser.add_argument("--families", help="comma-separated (default: every family in the suite)")
    rate_parser.add_argument("--band", default="0.1,0.7", help="solve rates counted as useful difficulty (default: %(default)s)")
    rate_parser.add_argument("--jobs-dir", default="jobs", help="where Harbor writes trial logs (default: %(default)s)")
    rate_parser.set_defaults(handler=solve_rate)

    convert_parser = commands.add_parser("convert", help="raw Claude Code / Codex / pi session logs -> ATIF")
    convert_parser.add_argument("sessions", nargs="+", help="session .jsonl files, or hf://datasets/<repo>[/<glob>]")
    convert_parser.add_argument("--format", required=True, choices=["claude-code", "codex", "pi"])
    convert_parser.add_argument("--limit", type=int, help="convert only the first N sessions")
    convert_parser.add_argument("--out", required=True)
    convert_parser.set_defaults(handler=convert)

    args = parser.parse_args()
    if getattr(args, "platform", None):
        docker.PLATFORM = args.platform
    args.handler(args)


if __name__ == "__main__":
    main()
