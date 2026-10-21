"""
Collect every finished benchmark in ./results/ into one comparison table.

Usage:
  python summarize.py                 # prints a Markdown table
  python summarize.py --csv out.csv   # also writes a CSV

Recovery % is measured against the BF16 original (the results folder whose
name starts with "Qwen3.6-35B-A3B-Qwen"). Run benchmark.sh on Qwen/Qwen3.6-35B-A3B
first to get that reference.
"""
import argparse
import csv
import glob
import json
import os

RESULTS = "results"
REFERENCE_PREFIX = "Qwen3.6-35B-A3B-Qwen"


def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def eval_score(folder, task_dir, task_prefix, metric_prefix):
    """Read the newest lm-eval results JSON for one task.

    Uses the overall (group) score, not a single subject: the group is the
    shortest task name, e.g. "mmlu_pro_llama" rather than "mmlu_pro_llama_biology".
    Groups are stored under "groups" in newer lm-eval versions, so check both.
    """
    f = latest(os.path.join(folder, task_dir, "*", "results_*.json"))
    if not f:
        return None
    with open(f) as fh:
        data = json.load(fh)
    entries = {**data.get("results", {}), **data.get("groups", {})}
    candidates = sorted((t for t in entries if t.startswith(task_prefix)), key=len)
    for task in candidates:
        for key, value in entries[task].items():
            if key.startswith(metric_prefix) and "stderr" not in key:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


def response_text(sample):
    """Extract the model's raw answer text from one lm-eval sample."""
    resps = sample.get("resps")
    while isinstance(resps, list) and resps:
        resps = resps[0]
    return resps if isinstance(resps, str) else ""


def empty_answers(folder, task_dir):
    """Count answers where the model only thought and never gave a final answer.

    Reads every subject file from the newest run, and counts each question once
    (lm-eval can write one line per answer filter).
    """
    files = glob.glob(os.path.join(folder, task_dir, "*", "samples_*.jsonl"))
    if not files:
        return None
    # File names end in _<timestamp>.jsonl; keep only the newest run.
    newest = max(f.rsplit("_", 1)[-1] for f in files)
    files = [f for f in files if f.endswith(newest)]
    seen = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (os.path.basename(f), sample.get("doc_id"))
                text = response_text(sample)
                is_empty = (not text.strip()) or "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER" in text
                seen[key] = seen.get(key, False) or is_empty
    return f"{sum(seen.values())}/{len(seen)}"


def speed(folder, concurrency):
    f = os.path.join(folder, f"speed_c{concurrency}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return float(json.load(fh).get("output_throughput", 0)) or None


def size(folder):
    f = os.path.join(folder, "size.txt")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return fh.read().split()[0]


def collect():
    rows = []
    for folder in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        if not os.path.isdir(folder):
            continue
        rows.append({
            "model": os.path.basename(folder),
            "size": size(folder),
            "mmlu_pro": eval_score(folder, "mmlu_pro", "mmlu_pro", "exact_match"),
            "gsm8k": eval_score(folder, "gsm8k", "gsm8k", "exact_match,flexible"),
            "tok_s_1": speed(folder, 1),
            "tok_s_16": speed(folder, 16),
            "empty_mmlu": empty_answers(folder, "mmlu_pro"),
            "empty_gsm8k": empty_answers(folder, "gsm8k"),
        })
    return rows


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def recovery(value, ref):
    if value is None or not ref:
        return "—"
    return f"{100 * value / ref:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = collect()
    if not rows:
        print(f"No results found in ./{RESULTS}/")
        return

    ref = next((r for r in rows if r["model"].startswith(REFERENCE_PREFIX)), None)
    ref_mmlu = ref["mmlu_pro"] if ref else None
    ref_gsm = ref["gsm8k"] if ref else None
    if not ref:
        print("Note: no BF16 reference found yet, so recovery % is blank.\n")

    header = ["Model", "Size", "MMLU-Pro", "Recovery", "GSM8K", "Recovery",
              "1-user tok/s", "16-user tok/s", "Empty (MMLU)", "Empty (GSM8K)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    table = []
    for r in rows:
        cells = [
            r["model"], fmt(r["size"]),
            fmt(r["mmlu_pro"]), recovery(r["mmlu_pro"], ref_mmlu),
            fmt(r["gsm8k"]), recovery(r["gsm8k"], ref_gsm),
            fmt(r["tok_s_1"], 1), fmt(r["tok_s_16"], 1),
            fmt(r["empty_mmlu"]), fmt(r["empty_gsm8k"]),
        ]
        table.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(table)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()