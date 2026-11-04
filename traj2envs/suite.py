"""Lay out kept tasks as one Hub dataset: a subset per family plus a Harbor registry.

  README.md               dataset card; its YAML `configs:` define one subset per family
  registry.json           one Harbor dataset per family
  <family>/manifest.parquet, <family>/tasks/<task_id>/
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_CARD = """---
license: other
tags: [rl-environment, harbor, agent-traces, traj2envs]
configs:
{subsets}
---

# RL tasks generated with traj2envs

Harbor tasks generated from ATIF agent trajectories. Each task records its source trajectory in
`task.toml` and passed a trial matrix in Docker (its oracle solution scores 1, doing nothing scores 0).

| Subset | Tasks |
|---|---|
{task_counts}

`registry.json` holds one Harbor dataset per subset. Tasks from the same trajectory can be
near-duplicates: split train/eval by `source_trace`. Check the source datasets' licenses before
making this repo public.
"""


def write_suite(out_dir: Path, kept_records_by_family: dict[str, list[dict]]) -> None:
    registry, subsets, task_counts = [], [], []
    for family, records in kept_records_by_family.items():
        if not records:
            continue
        registry.append({"name": f"traj2envs-{family}", "version": "0.1", "tasks": [
            {"name": r["task_id"], "path": f"{family}/tasks/{r['task_id']}"} for r in records]})
        manifest = pa.Table.from_pylist([{key: str(value) for key, value in r.items()} for r in records])
        pq.write_table(manifest, out_dir / family / "manifest.parquet")
        subsets.append(f"- config_name: {family}\n  data_files:\n  - split: train\n    path: {family}/manifest.parquet")
        task_counts.append(f"| `{family}` | {len(records)} |")
    (out_dir / "registry.json").write_text(json.dumps(registry, indent=2))
    (out_dir / "README.md").write_text(DATASET_CARD.format(subsets="\n".join(subsets), task_counts="\n".join(task_counts)))
