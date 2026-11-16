#!/usr/bin/env python
"""Generate YAML configs for missing multi-seed c0 runs (seed expansion).

Writes one file per run under ``configs/seed_expansion/`` and a queue manifest
``configs/seed_expansion/queue.json`` consumed by ``run_seed_expansion_queue.py``.

Conventions match existing ``results_fp16`` c0 manifests:
  N=24, T=0.8 (YAML; elicitation c0 also applies T=0.8 / top_p=0.95 at decode),
  generation.top_p left at base default 1.0 (elicitation overrides),
  max_new_tokens=1024, subsample_seed=1234, strategy=configured_sampling,
  elicitation.id=c0, out_dir=results_fp16.
"""
from __future__ import annotations

import argparse
import json
import statistics
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "seed_expansion"

MODELS: Dict[str, str] = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# (data_name, subset|None, subsample, paper_cell_tag)
DATASETS: Dict[str, Tuple[str, Optional[str], int, str]] = {
    "gsm8k": ("gsm8k", "main", 200, "gsm8k_baseline"),
    "math500": ("math500", None, 200, "math500_baseline"),
    "gsm_symbolic_main": ("gsm_symbolic", "main", 150, "gsm_symbolic"),
    "gsm_symbolic_p2": ("gsm_symbolic", "p2", 150, "gsm_symbolic"),
}

# 14 cells that form the C3 control grid once P0 seeds exist.
CONTROL_CELLS: List[Tuple[str, str]] = [
    ("qwen1.5b", "gsm8k"),
    ("qwen3b", "gsm8k"),
    ("qwen7b", "gsm8k"),
    ("qwen1.5b", "math500"),
    ("qwen3b", "math500"),
    ("qwen7b", "math500"),
    ("qwen1.5b", "gsm_symbolic_main"),
    ("qwen3b", "gsm_symbolic_main"),
    ("qwen7b", "gsm_symbolic_main"),
    ("qwen1.5b", "gsm_symbolic_p2"),
    ("qwen3b", "gsm_symbolic_p2"),
    ("qwen7b", "gsm_symbolic_p2"),
    ("llama3b", "gsm8k"),
    ("llama3b", "math500"),
]

# P0: unlock controls on cells that currently lack seeds {1,2}.
P0_CELLS: List[Tuple[str, str]] = [
    ("qwen1.5b", "gsm_symbolic_main"),
    ("qwen3b", "gsm_symbolic_main"),
    ("qwen7b", "gsm_symbolic_main"),
    ("qwen1.5b", "gsm_symbolic_p2"),
    ("qwen3b", "gsm_symbolic_p2"),
    ("qwen7b", "gsm_symbolic_p2"),
    ("llama3b", "gsm8k"),
    ("llama3b", "math500"),
]

# Fallback GPU-hour estimates from results_fp16 c0 manifests (mean per cell;
# math500/Qwen-3B uses median to down-weight the anomalous short s0).
EST_GPU_HOURS: Dict[Tuple[str, str], float] = {
    ("qwen1.5b", "gsm8k"): 2.079,
    ("qwen3b", "gsm8k"): 3.052,
    ("qwen7b", "gsm8k"): 4.253,
    ("qwen1.5b", "math500"): 4.459,
    ("qwen3b", "math500"): 6.703,  # median of {1.65, 6.70, 6.87}
    ("qwen7b", "math500"): 9.582,
    ("qwen1.5b", "gsm_symbolic_main"): 1.631,
    ("qwen3b", "gsm_symbolic_main"): 2.402,
    ("qwen7b", "gsm_symbolic_main"): 1.500,
    ("qwen1.5b", "gsm_symbolic_p2"): 3.203,
    ("qwen3b", "gsm_symbolic_p2"): 4.495,
    ("qwen7b", "gsm_symbolic_p2"): 2.556,
    ("llama3b", "gsm8k"): 3.180,
    ("llama3b", "math500"): 3.482,
}


@dataclass(frozen=True)
class SeedRun:
    priority: str  # p0 | p1
    model_key: str
    dataset_key: str
    seed: int
    est_gpu_hours: float


def _stem(model_key: str, dataset_key: str, seed: int) -> str:
    return f"{model_key}__{dataset_key}__c0__s{seed}"


def build_queue() -> List[SeedRun]:
    runs: List[SeedRun] = []
    for model_key, dataset_key in P0_CELLS:
        for seed in (1, 2):
            runs.append(
                SeedRun(
                    priority="p0",
                    model_key=model_key,
                    dataset_key=dataset_key,
                    seed=seed,
                    est_gpu_hours=EST_GPU_HOURS[(model_key, dataset_key)],
                )
            )
    for model_key, dataset_key in CONTROL_CELLS:
        for seed in (3, 4):
            runs.append(
                SeedRun(
                    priority="p1",
                    model_key=model_key,
                    dataset_key=dataset_key,
                    seed=seed,
                    est_gpu_hours=EST_GPU_HOURS[(model_key, dataset_key)],
                )
            )
    return runs


def render_yaml(run: SeedRun) -> str:
    model_path = MODELS[run.model_key]
    data_name, subset, subsample, paper_cell = DATASETS[run.dataset_key]
    subset_yaml = "null" if subset is None else subset
    return textwrap.dedent(
        f"""\
        defaults: ../base.yaml

        # Seed-expansion {run.priority.upper()}: c0 replicate seed={run.seed}.
        # Matches results_fp16 configured_sampling c0 conventions (N=24, T=0.8,
        # max_new_tokens=1024, subsample_seed=1234). Elicitation c0 applies
        # top_p=0.95 at decode; generation.top_p stays at base 1.0 for hash parity.
        # defaults path is relative to this file (configs/seed_expansion/).
        model:
          name_or_path: {model_path}
          backend: hf
          dtype: float16

        data:
          name: {data_name}
          split: test
          subset: {subset_yaml}
          subsample: {subsample}
          subsample_seed: 1234

        generation:
          max_new_tokens: 1024
          temperature: 0.8

        strategy:
          name: configured_sampling
          params:
            n_samples: 24
            batch_calls: true

        elicitation:
          id: c0

        runtime:
          seed: {run.seed}
          out_dir: results_fp16
          time_budget_hours: 11.0
          reserve_minutes: 10.0
          grade_after_run: true

        tags:
          paper_cell: {paper_cell}
          seed_expansion: {run.priority}
          seed_expansion_cell: {run.model_key}/{run.dataset_key}
        """
    )


def write_configs(
    runs: Sequence[SeedRun],
    out_dir: Path,
    *,
    clean: bool = True,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for old in out_dir.glob("*.yaml"):
            old.unlink()
        q = out_dir / "queue.json"
        if q.exists():
            q.unlink()

    entries: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        stem = _stem(run.model_key, run.dataset_key, run.seed)
        rel = f"configs/seed_expansion/{stem}.yaml"
        path = ROOT / rel
        path.write_text(render_yaml(run), encoding="utf-8")
        entry = {
            "order": idx,
            "priority": run.priority,
            "config": rel.replace("\\", "/"),
            "stem": stem,
            "model": MODELS[run.model_key],
            "dataset_key": run.dataset_key,
            "data": DATASETS[run.dataset_key][0],
            "subset": DATASETS[run.dataset_key][1],
            "seed": run.seed,
            "est_gpu_hours": round(run.est_gpu_hours, 4),
            "command": f"python -m src.runner --config {rel.replace(chr(92), '/')}",
        }
        entries.append(entry)
    (out_dir / "queue.json").write_text(
        json.dumps(
            {
                "description": "Seed-expansion c0 queue (P0 then P1)",
                "n_p0": sum(1 for e in entries if e["priority"] == "p0"),
                "n_p1": sum(1 for e in entries if e["priority"] == "p1"),
                "est_gpu_hours_p0": round(
                    sum(e["est_gpu_hours"] for e in entries if e["priority"] == "p0"), 2
                ),
                "est_gpu_hours_p1": round(
                    sum(e["est_gpu_hours"] for e in entries if e["priority"] == "p1"), 2
                ),
                "est_gpu_hours_total": round(sum(e["est_gpu_hours"] for e in entries), 2),
                "runs": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return entries


def refresh_estimates_from_manifests(results_dir: Path) -> None:
    """Optional: overwrite EST_GPU_HOURS from local manifests (best-effort)."""
    if not results_dir.is_dir():
        return
    buckets: Dict[Tuple[str, str], List[float]] = {}
    inv_models = {v.split("/")[-1]: k for k, v in MODELS.items()}
    inv_data = {
        ("gsm8k", "main"): "gsm8k",
        ("math500", None): "math500",
        ("math500", "-"): "math500",
        ("gsm_symbolic", "main"): "gsm_symbolic_main",
        ("gsm_symbolic", "p2"): "gsm_symbolic_p2",
    }
    for man in results_dir.glob("*configured_sampling__c0*/manifest.json"):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("config_hash") == "685e9b58e0b8":
            continue  # alternate math500/3B hash; not the seed-series family
        cfg = m.get("config") or {}
        model = (cfg.get("model") or {}).get("name_or_path", "").split("/")[-1]
        data = (cfg.get("data") or {}).get("name")
        subset = (cfg.get("data") or {}).get("subset")
        mk = inv_models.get(model)
        dk = inv_data.get((data, subset if subset is not None else None))
        if dk is None and data == "math500":
            dk = "math500"
        if mk is None or dk is None:
            continue
        h = m.get("gpu_hours")
        if h is None and m.get("elapsed_seconds") is not None:
            h = float(m["elapsed_seconds"]) / 3600.0
        if h is None:
            continue
        buckets.setdefault((mk, dk), []).append(float(h))
    for key, vals in buckets.items():
        if key[1] == "math500" and key[0] == "qwen3b" and len(vals) >= 3:
            EST_GPU_HOURS[key] = float(statistics.median(vals))
        else:
            EST_GPU_HOURS[key] = float(statistics.mean(vals))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for generated YAMLs (default: configs/seed_expansion)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results_fp16",
        help="Existing results for GPU-hour estimates",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing YAMLs in out-dir first",
    )
    args = p.parse_args()
    refresh_estimates_from_manifests(args.results_dir)
    runs = build_queue()
    entries = write_configs(runs, args.out_dir, clean=not args.no_clean)
    n0 = sum(1 for e in entries if e["priority"] == "p0")
    n1 = sum(1 for e in entries if e["priority"] == "p1")
    h0 = sum(e["est_gpu_hours"] for e in entries if e["priority"] == "p0")
    h1 = sum(e["est_gpu_hours"] for e in entries if e["priority"] == "p1")
    print(f"wrote {len(entries)} configs -> {args.out_dir}")
    print(f"  P0: {n0} runs, ~{h0:.1f} GPU-h")
    print(f"  P1: {n1} runs, ~{h1:.1f} GPU-h")
    print(f"  total: {len(entries)} runs, ~{h0 + h1:.1f} GPU-h")
    print(f"  queue: {args.out_dir / 'queue.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
