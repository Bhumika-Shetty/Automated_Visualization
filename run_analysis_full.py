"""
Full analysis script for AI Visualization Evaluation Pipeline.
8 models, 8 datasets, 2 prompt types, 30 runs = 3,840 valid runs.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

results_dir = Path("outputs/results")

MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "deepseek-chat",
    "gemini-2_5-flash",
    "gemini-2_5-pro",
    "gpt-4o",
    "gpt-4o-mini",
    "llama-3_1-8b-instant",
    "qwen_qwen3-32b",
]

MODEL_DISPLAY = {
    "claude-opus-4-5": "Claude Opus 4-5",
    "claude-sonnet-4-5": "Claude Sonnet 4-5",
    "deepseek-chat": "DeepSeek Chat",
    "gemini-2_5-flash": "Gemini 2.5 Flash",
    "gemini-2_5-pro": "Gemini 2.5 Pro",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-3_1-8b-instant": "Llama 3.1 8B",
    "qwen_qwen3-32b": "Qwen3 32B",
}

PROVIDER = {
    "claude-opus-4-5": "Anthropic",
    "claude-sonnet-4-5": "Anthropic",
    "deepseek-chat": "DeepSeek",
    "gemini-2_5-flash": "Google",
    "gemini-2_5-pro": "Google",
    "gpt-4o": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "llama-3_1-8b-instant": "Groq (OSS)",
    "qwen_qwen3-32b": "Groq (OSS)",
}

DATASETS = ["titanic", "iris", "sleep_health", "customer_segments", "credit_fraud", "dogs_zurich", "cameras", "coffee_sales"]
KNOWN_DATASETS = ["titanic", "iris", "sleep_health", "customer_segments", "credit_fraud"]
UNKNOWN_DATASETS = ["dogs_zurich", "cameras", "coffee_sales"]
PROMPTS = ["rich", "poor"]


def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def safe_std(vals):
    v = [x for x in vals if x is not None]
    return float(np.std(v)) if len(v) > 1 else None


def safe_pct(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) * 100 if v else None


def load_all_data():
    all_data = []
    for model in MODELS:
        for f in results_dir.glob(f"{model}_*.json"):
            try:
                d = json.loads(f.read_text())
                if d.get("error"):
                    continue
                m = d.get("metrics", {})
                if not m:
                    continue
                stem = f.stem
                found = False
                for ds in DATASETS:
                    for pt in PROMPTS:
                        if f"_{ds}_{pt}_" in stem:
                            all_data.append({
                                "model": model,
                                "display": MODEL_DISPLAY[model],
                                "provider": PROVIDER[model],
                                "dataset": ds,
                                "prompt_type": pt,
                                "run_id": d.get("run_id"),
                                "execution_success": bool(m.get("execution_success", False)),
                                "fidelity_score": m.get("fidelity_score") or 0,
                                "hallucination_rate": m.get("hallucination_rate") or 0,
                                "color_delta_e": m.get("color_delta_e"),
                                "visual_entropy": m.get("visual_entropy"),
                                "labeling_completeness": m.get("labeling_completeness") or 0,
                                "code_complexity": m.get("code_complexity") or 0,
                                "generation_latency": m.get("generation_latency") or 0,
                                "cost_per_run": m.get("cost_per_run") or 0,
                                "fix_applied": bool(d.get("fix_applied")),
                                "dataset_type": "known" if ds in KNOWN_DATASETS else "unknown",
                            })
                            found = True
                            break
                    if found:
                        break
            except Exception:
                continue
    return all_data


def get_rows(data, model=None, dataset=None, prompt=None, dataset_type=None):
    rows = data
    if model:
        rows = [r for r in rows if r["model"] == model]
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    if prompt:
        rows = [r for r in rows if r["prompt_type"] == prompt]
    if dataset_type:
        rows = [r for r in rows if r["dataset_type"] == dataset_type]
    return rows


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    data = load_all_data()
    print(f"Total valid runs loaded: {len(data)}")

    # ─────────────────────────────────────────────
    # 1. OVERALL MODEL SUMMARY
    # ─────────────────────────────────────────────
    print_section("1. OVERALL MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<22} {'Provider':<14} {'N':>5} {'Success%':>9} {'Fidelity':>9} {'Halluc%':>8} {'Label%':>7} {'Latency':>8} {'Cost($)':>9} {'Complexity':>11} {'Entropy':>8}")
    print("-" * 110)

    model_stats = {}
    for model in MODELS:
        rows = get_rows(data, model=model)
        if not rows:
            continue
        succ = safe_pct([r["execution_success"] for r in rows])
        fid = safe_mean([r["fidelity_score"] for r in rows])
        hall = safe_pct([r["hallucination_rate"] for r in rows])
        lab = safe_pct([r["labeling_completeness"] for r in rows])
        lat = safe_mean([r["generation_latency"] for r in rows])
        cost = safe_mean([r["cost_per_run"] for r in rows])
        comp = safe_mean([r["code_complexity"] for r in rows])
        ent = safe_mean([r["visual_entropy"] for r in rows if r["visual_entropy"]])
        model_stats[model] = dict(succ=succ, fid=fid, hall=hall, lab=lab, lat=lat, cost=cost, comp=comp, ent=ent, n=len(rows))
        print(f"{MODEL_DISPLAY[model]:<22} {PROVIDER[model]:<14} {len(rows):>5} {succ:>8.1f}% {fid:>9.2f} {hall:>7.1f}% {lab:>6.1f}% {lat:>7.1f}s {cost:>9.5f} {comp:>11.2f} {ent if ent else 0:>8.3f}")

    # ─────────────────────────────────────────────
    # 2. PROMPT SENSITIVITY (RICH vs POOR)
    # ─────────────────────────────────────────────
    print_section("2. PROMPT SENSITIVITY — Rich vs Poor")
    print(f"{'Model':<22} {'Rich%':>7} {'Poor%':>7} {'Gap':>7} {'Direction':<15} {'PSI':>6}")
    print("-" * 75)

    for model in MODELS:
        rich = get_rows(data, model=model, prompt="rich")
        poor = get_rows(data, model=model, prompt="poor")
        if not rich or not poor:
            continue
        rich_pct = safe_pct([r["execution_success"] for r in rich])
        poor_pct = safe_pct([r["execution_success"] for r in poor])
        gap = rich_pct - poor_pct
        direction = "Rich > Poor" if gap > 0 else "Poor > Rich"
        psi = abs(gap)
        print(f"{MODEL_DISPLAY[model]:<22} {rich_pct:>6.1f}% {poor_pct:>6.1f}% {gap:>+6.1f}% {direction:<15} {psi:>5.1f}%")

    # ─────────────────────────────────────────────
    # 3. KNOWN vs UNKNOWN DATASETS
    # ─────────────────────────────────────────────
    print_section("3. KNOWN vs UNCONVENTIONAL DATASETS")
    print(f"{'Model':<22} {'Known%':>8} {'Unknown%':>9} {'Gap':>7}")
    print("-" * 55)

    for model in MODELS:
        known = get_rows(data, model=model, dataset_type="known")
        unknown = get_rows(data, model=model, dataset_type="unknown")
        if not known or not unknown:
            continue
        k_pct = safe_pct([r["execution_success"] for r in known])
        u_pct = safe_pct([r["execution_success"] for r in unknown])
        gap = k_pct - u_pct
        print(f"{MODEL_DISPLAY[model]:<22} {k_pct:>7.1f}% {u_pct:>8.1f}% {gap:>+6.1f}%")

    # ─────────────────────────────────────────────
    # 4. PER-DATASET SUCCESS RATES
    # ─────────────────────────────────────────────
    print_section("4. SUCCESS RATE BY DATASET (all models combined)")
    print(f"{'Dataset':<22} {'Type':<10} {'N':>5} {'Success%':>9} {'Fidelity':>9} {'Halluc%':>8}")
    print("-" * 70)

    for ds in DATASETS:
        rows = get_rows(data, dataset=ds)
        dtype = "Known" if ds in KNOWN_DATASETS else "Unknown"
        succ = safe_pct([r["execution_success"] for r in rows])
        fid = safe_mean([r["fidelity_score"] for r in rows])
        hall = safe_pct([r["hallucination_rate"] for r in rows])
        print(f"{ds:<22} {dtype:<10} {len(rows):>5} {succ:>8.1f}% {fid:>9.2f} {hall:>7.1f}%")

    # ─────────────────────────────────────────────
    # 5. PROVIDER COMPARISON
    # ─────────────────────────────────────────────
    print_section("5. PROVIDER-LEVEL AGGREGATES")
    print(f"{'Provider':<16} {'N':>6} {'Success%':>9} {'Fidelity':>9} {'Halluc%':>8} {'Latency':>8} {'Cost($)':>9}")
    print("-" * 75)

    providers = ["OpenAI", "Anthropic", "Google", "DeepSeek", "Groq (OSS)"]
    for prov in providers:
        rows = [r for r in data if r["provider"] == prov]
        if not rows:
            continue
        succ = safe_pct([r["execution_success"] for r in rows])
        fid = safe_mean([r["fidelity_score"] for r in rows])
        hall = safe_pct([r["hallucination_rate"] for r in rows])
        lat = safe_mean([r["generation_latency"] for r in rows])
        cost = safe_mean([r["cost_per_run"] for r in rows])
        print(f"{prov:<16} {len(rows):>6} {succ:>8.1f}% {fid:>9.2f} {hall:>7.1f}% {lat:>7.1f}s {cost:>9.5f}")

    # ─────────────────────────────────────────────
    # 6. CROSS-RUN CONSISTENCY (CV of success per model-dataset combo)
    # ─────────────────────────────────────────────
    print_section("6. CROSS-RUN CONSISTENCY (Coefficient of Variation of fidelity score)")
    print(f"{'Model':<22} {'CV_fidelity':>12} {'CV_halluc':>10} {'CV_latency':>11}")
    print("-" * 60)

    for model in MODELS:
        rows = get_rows(data, model=model)
        fid_vals = [r["fidelity_score"] for r in rows]
        hall_vals = [r["hallucination_rate"] for r in rows]
        lat_vals = [r["generation_latency"] for r in rows]

        def cv(vals):
            m = safe_mean(vals)
            s = safe_std(vals)
            return s / m if m and m > 0 else None

        cv_fid = cv(fid_vals)
        cv_hall = cv(hall_vals)
        cv_lat = cv(lat_vals)
        print(f"{MODEL_DISPLAY[model]:<22} {cv_fid if cv_fid else 0:>11.3f} {cv_hall if cv_hall else 0:>9.3f} {cv_lat if cv_lat else 0:>10.3f}")

    # ─────────────────────────────────────────────
    # 7. VISUAL QUALITY (color_delta_e + visual_entropy) for successful runs
    # ─────────────────────────────────────────────
    print_section("7. VISUAL QUALITY METRICS (successful runs only)")
    print(f"{'Model':<22} {'N_success':>10} {'Color_DeltaE':>13} {'Visual_Entropy':>15} {'Labeling%':>10}")
    print("-" * 75)

    for model in MODELS:
        rows = [r for r in get_rows(data, model=model) if r["execution_success"]]
        if not rows:
            continue
        cde = safe_mean([r["color_delta_e"] for r in rows if r["color_delta_e"]])
        ent = safe_mean([r["visual_entropy"] for r in rows if r["visual_entropy"]])
        lab = safe_pct([r["labeling_completeness"] for r in rows])
        print(f"{MODEL_DISPLAY[model]:<22} {len(rows):>10} {cde if cde else 0:>12.3f} {ent if ent else 0:>14.3f} {lab:>9.1f}%")

    # ─────────────────────────────────────────────
    # 8. COST vs QUALITY
    # ─────────────────────────────────────────────
    print_section("8. COST-EFFICIENCY (cost per successful visualization)")
    print(f"{'Model':<22} {'Avg_cost':>10} {'Success%':>9} {'Cost/Success':>13}")
    print("-" * 60)

    for model in MODELS:
        rows = get_rows(data, model=model)
        succ = safe_pct([r["execution_success"] for r in rows]) / 100
        avg_cost = safe_mean([r["cost_per_run"] for r in rows])
        cost_per_succ = avg_cost / succ if succ and succ > 0 and avg_cost else 0
        print(f"{MODEL_DISPLAY[model]:<22} {avg_cost if avg_cost else 0:>10.6f} {succ*100 if succ else 0:>8.1f}% {cost_per_succ:>13.6f}")

    # ─────────────────────────────────────────────
    # 9. DETAILED PER-MODEL + PER-DATASET BREAKDOWN
    # ─────────────────────────────────────────────
    print_section("9. PER-MODEL × PER-DATASET SUCCESS RATES")
    header = f"{'Model':<22}"
    for ds in DATASETS:
        header += f" {ds[:8]:>9}"
    print(header)
    print("-" * (22 + 9 * len(DATASETS)))

    for model in MODELS:
        row_str = f"{MODEL_DISPLAY[model]:<22}"
        for ds in DATASETS:
            rows = get_rows(data, model=model, dataset=ds)
            succ = safe_pct([r["execution_success"] for r in rows]) if rows else 0
            row_str += f" {succ:>8.0f}%"
        print(row_str)

    # ─────────────────────────────────────────────
    # 10. KEY FINDINGS SUMMARY
    # ─────────────────────────────────────────────
    print_section("10. KEY FINDINGS SUMMARY")

    # Best model overall
    best_model = max(model_stats, key=lambda m: model_stats[m]["succ"])
    worst_model = min(model_stats, key=lambda m: model_stats[m]["succ"])
    print(f"• Best overall success rate: {MODEL_DISPLAY[best_model]} ({model_stats[best_model]['succ']:.1f}%)")
    print(f"• Worst overall success rate: {MODEL_DISPLAY[worst_model]} ({model_stats[worst_model]['succ']:.1f}%)")
    print(f"• Gap between best and worst: {model_stats[best_model]['succ'] - model_stats[worst_model]['succ']:.1f}pp")

    # Fastest model
    fastest = min(model_stats, key=lambda m: model_stats[m]["lat"] or 999)
    slowest = max(model_stats, key=lambda m: model_stats[m]["lat"] or 0)
    print(f"• Fastest model: {MODEL_DISPLAY[fastest]} ({model_stats[fastest]['lat']:.1f}s avg latency)")
    print(f"• Slowest model: {MODEL_DISPLAY[slowest]} ({model_stats[slowest]['lat']:.1f}s avg latency)")

    # Cheapest
    cheapest = min(model_stats, key=lambda m: model_stats[m]["cost"] or 999)
    priciest = max(model_stats, key=lambda m: model_stats[m]["cost"] or 0)
    print(f"• Cheapest model: {MODEL_DISPLAY[cheapest]} (${model_stats[cheapest]['cost']:.6f}/run)")
    print(f"• Most expensive model: {MODEL_DISPLAY[priciest]} (${model_stats[priciest]['cost']:.6f}/run)")

    # Prompt sensitivity
    print()
    print("• Prompt sensitivity reversal check:")
    for model in MODELS:
        rich = get_rows(data, model=model, prompt="rich")
        poor = get_rows(data, model=model, prompt="poor")
        if not rich or not poor:
            continue
        r = safe_pct([x["execution_success"] for x in rich])
        p = safe_pct([x["execution_success"] for x in poor])
        direction = "Rich > Poor" if r > p else "Poor > Rich"
        print(f"  {MODEL_DISPLAY[model]:<22}: Rich={r:.1f}% Poor={p:.1f}% → {direction} (gap={abs(r-p):.1f}pp)")

    # Known vs unknown
    print()
    print("• Known vs Unconventional dataset performance:")
    for model in MODELS:
        known = get_rows(data, model=model, dataset_type="known")
        unknown = get_rows(data, model=model, dataset_type="unknown")
        if not known or not unknown:
            continue
        k = safe_pct([x["execution_success"] for x in known])
        u = safe_pct([x["execution_success"] for x in unknown])
        print(f"  {MODEL_DISPLAY[model]:<22}: Known={k:.1f}% Unknown={u:.1f}% (gap={k-u:+.1f}pp)")


if __name__ == "__main__":
    main()
