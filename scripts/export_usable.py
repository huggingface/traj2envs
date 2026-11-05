"""Copy scored, published, usable tasks (attempts not all 1.0, not all 0) from suites/shard-* into one dataset folder.

  python scripts/export_usable.py OUT_DIR          <family>/tasks/<task_id>, one subset per family (traj2envs layout)
  python scripts/export_usable.py OUT_DIR --flat   tasks/<task_id>, one manifest with a `family` column

Each exported task.toml gets a top-level `difficulty` from its solve rate (hard < 0.3 <= medium < 0.6 <= easy).
OUT_DIR/sources.json lists the source datasets and trajectory files (to block fetching them while training with
internet), and OUT_DIR/images.txt the distinct images the tasks pin.
"""
import glob, json, shutil, sys, tomllib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from traj2envs.suite import write_suite


def difficulty(solve_rate: float) -> str:
    return "hard" if solve_rate < 0.3 else "medium" if solve_rate < 0.6 else "easy"


out, flat = Path(sys.argv[1]), "--flat" in sys.argv[2:]
shutil.rmtree(out, ignore_errors=True)
by_family, skipped, images, sources = {}, {"saturated": 0, "never solved": 0, "not published": 0}, set(), set()
for rates_path in sorted(glob.glob("suites/shard-*/solve_rates.json")):
    shard = Path(rates_path).parent
    manifests = {f.parent.name: {r["task_id"]: r for r in pq.read_table(f).to_pylist()}
                 for f in shard.glob("*/manifest.parquet")}
    for t in json.load(open(rates_path))["tasks"]:
        scored = [x for x in t["rewards"] if x is not None]
        task_dir = shard / t["family"] / "tasks" / t["task_id"]
        config = tomllib.loads((task_dir / "task.toml").read_text())
        if not scored or all(x == 1.0 for x in scored):
            skipped["saturated"] += 1; continue
        if all(x == 0 for x in scored):
            skipped["never solved"] += 1; continue
        if "docker_image" not in config["environment"]:
            skipped["not published"] += 1; continue
        destination = out / ("tasks" if flat else Path(t["family"]) / "tasks") / t["task_id"]
        if destination.exists():
            sys.exit(f"task id {t['task_id']} appears twice")
        shutil.copytree(task_dir, destination)
        config = {"difficulty": difficulty(t["solve_rate"]), **config}
        (destination / "task.toml").write_text(tomli_w.dumps(config))
        images.add(config["environment"]["docker_image"])
        sources.add(config["metadata"].get("source_trace", ""))
        record = {**manifests.get(t["family"], {}).get(t["task_id"], {"task_id": t["task_id"]}),
                  "family": t["family"], "shard": shard.name, "solve_rate": t["solve_rate"],
                  "difficulty": config["difficulty"], "rewards": json.dumps(t["rewards"])}
        by_family.setdefault(t["family"], []).append(record)

files = sorted(s for s in sources if s)
datasets = sorted({"/".join(s.removeprefix("hf://datasets/").split("/")[:2]) for s in files if s.startswith("hf://datasets/")})
(out / "sources.json").write_text(json.dumps({"datasets": datasets, "files": files}, indent=1))
(out / "images.txt").write_text("".join(f"{image}\n" for image in sorted(images)))
records = [r for family_records in by_family.values() for r in family_records]
if flat:
    pq.write_table(pa.Table.from_pylist([{k: str(v) for k, v in r.items()} for r in records]), out / "manifest.parquet")
    (out / "registry.json").write_text(json.dumps([{"name": "traj2envs", "version": "0.1", "tasks": [
        {"name": r["task_id"], "path": f"tasks/{r['task_id']}"} for r in records]}], indent=2))
    (out / "README.md").write_text(
        "---\nlicense: other\ntags: [rl-environment, harbor, agent-traces, traj2envs]\nconfigs:\n- config_name: default\n"
        "  data_files:\n  - split: train\n    path: manifest.parquet\n---\n\n# RL tasks generated with traj2envs\n\n"
        f"{len(records)} Harbor tasks under `tasks/`, each with its environment image pinned in `task.toml` "
        "(`[environment].docker_image`, with opencode preinstalled), a top-level `difficulty` and its solve rate in "
        "`[metadata]`. `manifest.parquet` lists them with their family. `sources.json` lists the trajectories they "
        "were built from: block fetching them when training with internet. Tasks from the same trajectory can be "
        "near-duplicates: split train/eval by `source_trace`.\n")
else:
    write_suite(out, by_family)
counts = {level: sum(r["difficulty"] == level for r in records) for level in ("hard", "medium", "easy")}
print(f"{len(records)} tasks", {family: len(rs) for family, rs in by_family.items()}, counts, "skipped:", skipped)
print(f"{len(images)} images, {len(datasets)} source datasets, {len(files)} source files")
