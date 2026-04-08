"""
analysis.py - Load all result JSONs and compute summary statistics.
Exports summary_table.csv and prompt_sensitivity.csv to outputs/
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_results(results_dir: str = "outputs/results") -> pd.DataFrame:
    results_path = Path(results_dir)
    records = []
    for f in results_path.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("metrics"):
                row = {
                    "model": data["model"],
                    "dataset": data["dataset"],
                    "prompt_type": data["prompt_type"],
                    "run_id": data["run_id"],
                    **data["metrics"],
                }
                records.append(row)
        except Exception:
            continue
    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary stats."""
    summary = df.groupby("model").agg(
        success_rate=("execution_success", "mean"),
        mean_fidelity=("fidelity_score", "mean"),
        mean_hallucination=("hallucination_rate", "mean"),
        mean_labeling=("labeling_completeness", "mean"),
        mean_latency=("generation_latency", "mean"),
        mean_cost=("cost_per_run", "mean"),
        mean_complexity=("code_complexity", "mean"),
        mean_entropy=("visual_entropy", lambda x: x.dropna().mean()),
        total_runs=("run_id", "count"),
    ).reset_index()

    summary["success_rate"] = (summary["success_rate"] * 100).round(1)
    summary["mean_fidelity"] = summary["mean_fidelity"].round(2)
    summary["mean_hallucination"] = summary["mean_hallucination"].round(4)
    summary["mean_labeling"] = (summary["mean_labeling"] * 100).round(1)
    summary["mean_latency"] = summary["mean_latency"].round(2)
    summary["mean_cost"] = summary["mean_cost"].round(6)
    return summary.sort_values("success_rate", ascending=False)


def compute_prompt_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Rich vs poor success rate per model."""
    pivot = df.groupby(["model", "prompt_type"])["execution_success"].mean().unstack()
    pivot = pivot * 100
    pivot.columns = [f"{c}_success_pct" for c in pivot.columns]
    pivot["gap_rich_minus_poor"] = pivot.get("rich_success_pct", 0) - pivot.get("poor_success_pct", 0)
    pivot["prompt_sensitivity_index"] = pivot["gap_rich_minus_poor"].abs()
    return pivot.reset_index().sort_values("prompt_sensitivity_index", ascending=False)


def compute_provider_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Provider-level aggregates."""
    def get_provider(model):
        if model.startswith("gpt"):
            return "OpenAI"
        elif model.startswith("claude"):
            return "Anthropic"
        elif model.startswith("gemini"):
            return "Google"
        return "Other"

    df = df.copy()
    df["provider"] = df["model"].apply(get_provider)
    return df.groupby("provider").agg(
        success_rate=("execution_success", lambda x: round(x.mean() * 100, 1)),
        mean_fidelity=("fidelity_score", lambda x: round(x.mean(), 2)),
        mean_latency=("generation_latency", lambda x: round(x.mean(), 2)),
        total_runs=("run_id", "count"),
    ).reset_index().sort_values("success_rate", ascending=False)


def run_analysis(results_dir: str = "outputs/results", output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        print("No results found.")
        return

    print(f"Loaded {len(df)} results.")

    summary = compute_summary(df)
    summary.to_csv(out / "summary_table.csv", index=False)
    print("\n=== Model Summary ===")
    print(summary.to_string(index=False))

    ps = compute_prompt_sensitivity(df)
    ps.to_csv(out / "prompt_sensitivity.csv", index=False)
    print("\n=== Prompt Sensitivity ===")
    print(ps.to_string(index=False))

    prov = compute_provider_summary(df)
    prov.to_csv(out / "provider_summary.csv", index=False)
    print("\n=== Provider Summary ===")
    print(prov.to_string(index=False))

    return {"summary": summary, "prompt_sensitivity": ps, "provider_summary": prov}


if __name__ == "__main__":
    run_analysis()
