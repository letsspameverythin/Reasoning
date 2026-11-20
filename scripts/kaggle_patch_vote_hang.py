"""One-shot Kaggle patch: stop sympy hangs in majority_vote (GPU util=0 stall).

Run from the harness root (/kaggle/working/harness), then re-launch the same
runner command. Resume is safe — completed rows in results.jsonl are skipped.

    %cd /kaggle/working/harness
    !python scripts/kaggle_patch_vote_hang.py
    !python -m src.runner --config configs/seed_expansion/qwen1.5b__math500__c0__s3.yaml
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "src" / "answers.py"


def main() -> None:
    text = ANSWERS.read_text(encoding="utf-8")
    if "strict: bool = False" in text and "cluster[\"key\"] == key" in text:
        print("already patched:", ANSWERS)
        return

    needle = '''    for raw in answers:
        if raw is None or not str(raw).strip():
            n_none += 1
            continue
        value = str(raw)
        placed = False
        for cluster in clusters:
            if answers_equivalent(value, cluster["key"], answer_type, choices):
                cluster["count"] += 1
                cluster["members"].append(value)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "key": normalize_answer(value, answer_type) or value,
                    "representative": value,
                    "count": 1,
                    "members": [value],
                }
            )'''

    replacement = '''    for raw in answers:
        if raw is None or not str(raw).strip():
            n_none += 1
            continue
        value = str(raw)
        key = normalize_answer(value, answer_type) or value
        placed = False
        for cluster in clusters:
            if cluster["key"] == key:
                cluster["count"] += 1
                cluster["members"].append(value)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "key": key,
                    "representative": value,
                    "count": 1,
                    "members": [value],
                }
            )'''

    if needle not in text:
        raise SystemExit(
            "majority_vote loop not found — upload the fixed harness instead of "
            "this patch script"
        )
    ANSWERS.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print("patched majority_vote to normalize-only clustering:", ANSWERS)
    print("kill the stuck runner (Interrupt), then re-run the same --config command")


if __name__ == "__main__":
    main()
