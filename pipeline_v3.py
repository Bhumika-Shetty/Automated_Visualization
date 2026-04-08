"""
pipeline_v3.py - Main automated evaluation pipeline.
Runs all model × dataset × prompt_type × run combinations.
Resumable: skips completed runs.
"""

import os
import sys
import json
import re
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from api_adapters import call_model
from datasets import load_dataset
from prompts_v2 import get_prompt
from metrics_v2 import calculate_all_metrics

DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "deepseek-chat",
    "qwen-qwq-32b",
    "llama-3.1-8b-instant",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

DEFAULT_DATASETS = ["titanic", "iris", "sleep_health", "customer_segments", "credit_fraud", "dogs_zurich", "cameras", "coffee_sales"]
PROMPT_TYPES = ["rich", "poor"]


def extract_code(response: str) -> str:
    """Strip markdown fences and extract Python code."""
    # Try ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return as-is if it contains imports
    if "import " in response:
        return response.strip()
    return response.strip()


def run_code_isolated(code: str, data_dir: Path, timeout: int = 60) -> tuple:
    """
    Execute code in an isolated temp directory.
    Returns (success, png_bytes or None, stdout, stderr, fix_applied).
    """
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Copy data files into temp dir under data/
        data_tmpdir = tmpdir / "data"
        data_tmpdir.mkdir()
        for csv in data_dir.glob("*.csv"):
            shutil.copy(csv, data_tmpdir / csv.name)

        output_png = tmpdir / "output.png"

        def patch_and_write(c: str) -> Path:
            """Patch paths in code and write to temp script."""
            p = c
            # Fix output path
            p = p.replace("'output.png'", f"'{output_png}'")
            p = p.replace('"output.png"', f'"{output_png}"')
            # Fix data paths — only replace relative data/ references, not absolute ones
            for csv in data_dir.glob("*.csv"):
                rel = f"data/{csv.name}"
                abs_path = str(data_tmpdir / csv.name)
                if rel in p:
                    p = p.replace(rel, abs_path)
                elif csv.name in p and abs_path not in p:
                    # bare filename reference (e.g. pd.read_csv('titanic.csv'))
                    p = p.replace(f"'{csv.name}'", f"'{abs_path}'")
                    p = p.replace(f'"{csv.name}"', f'"{abs_path}"')
            script = tmpdir / "viz_script.py"
            script.write_text(p)
            return script

        script = patch_and_write(code)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout, cwd=str(tmpdir),
        )

        # Check if any png was produced (partial success)
        if not output_png.exists():
            pngs = list(tmpdir.glob("*.png"))
            if pngs:
                output_png = pngs[0]

        success = output_png.exists() and output_png.stat().st_size > 0
        fix_applied = None

        # --- Auto-fix: column name case mismatch ---
        if not success and result.stderr:
            stderr = result.stderr
            # Detect KeyError or column not found errors
            col_error = (
                "KeyError" in stderr or
                "column" in stderr.lower() or
                "not in index" in stderr.lower()
            )
            if col_error:
                # Try to fix: read actual columns from dataset and replace mismatched ones
                try:
                    import pandas as pd
                    import re as _re
                    # Find which dataset this is
                    for csv in data_dir.glob("*.csv"):
                        df = pd.read_csv(csv)
                        actual_cols = list(df.columns)
                        fixed_code = code
                        # Build a lower-case lookup
                        col_lower = {c.lower(): c for c in actual_cols}
                        # Replace quoted strings in code that look like wrong-case column names
                        def replace_col(m):
                            q, val, q2 = m.group(1), m.group(2), m.group(3)
                            fixed = col_lower.get(val.lower())
                            if fixed and fixed != val:
                                return f"{q}{fixed}{q2}"
                            return m.group(0)
                        fixed_code = _re.sub(r"(['\"])([A-Za-z_][A-Za-z0-9_ ]*)(['\"])", replace_col, fixed_code)
                        if fixed_code != code:
                            fix_applied = f"Auto-fixed column name case mismatch. Original error: {stderr[:200]}"
                            script2 = patch_and_write(fixed_code)
                            result2 = subprocess.run(
                                [sys.executable, str(script2)],
                                capture_output=True, text=True, timeout=timeout, cwd=str(tmpdir),
                            )
                            if not output_png.exists():
                                pngs = list(tmpdir.glob("*.png"))
                                if pngs:
                                    output_png = pngs[0]
                            success = output_png.exists() and output_png.stat().st_size > 0
                            result = result2  # use fixed run's stderr/stdout
                            break
                except Exception:
                    pass

        png_bytes = output_png.read_bytes() if success else None
        return success, png_bytes, result.stdout, result.stderr, fix_applied

    except subprocess.TimeoutExpired:
        return False, None, "", "Timeout after 60s", None
    except Exception as e:
        return False, None, "", str(e), None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class Pipeline:
    def __init__(
        self,
        models=None,
        datasets=None,
        num_runs=30,
        output_dir="outputs",
    ):
        self.models = models or DEFAULT_MODELS
        self.datasets = datasets or DEFAULT_DATASETS
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.images_dir = self.output_dir / "images"
        self.data_dir = Path(__file__).parent / "data"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def result_path(self, model, dataset, prompt_type, run_id):
        safe_model = model.replace("/", "_").replace(".", "_")
        return self.results_dir / f"{safe_model}_{dataset}_{prompt_type}_{run_id:03d}.json"

    def image_path(self, model, dataset, prompt_type, run_id):
        safe_model = model.replace("/", "_").replace(".", "_")
        return self.images_dir / f"{safe_model}_{dataset}_{prompt_type}_{run_id:03d}.png"

    def run(self):
        total = len(self.models) * len(self.datasets) * len(PROMPT_TYPES) * self.num_runs
        done = 0
        skipped = 0

        print(f"\n{'='*70}")
        print(f"AI Visualization Evaluation Pipeline v3")
        print(f"Models: {len(self.models)} | Datasets: {len(self.datasets)} | "
              f"Prompt types: 2 | Runs: {self.num_runs}")
        print(f"Total runs: {total}")
        print(f"{'='*70}\n")

        _, meta_cache = {}, {}

        for model in self.models:
            for dataset in self.datasets:
                # Pre-load dataset columns for metrics
                try:
                    df, meta = load_dataset(dataset)
                    columns = meta["columns"]
                except Exception as e:
                    print(f"  [ERROR] Could not load dataset {dataset}: {e}")
                    continue

                for prompt_type in PROMPT_TYPES:
                    try:
                        prompt = get_prompt(dataset, prompt_type)
                    except Exception as e:
                        print(f"  [ERROR] Could not get prompt {dataset}/{prompt_type}: {e}")
                        continue

                    for run_id in range(1, self.num_runs + 1):
                        rpath = self.result_path(model, dataset, prompt_type, run_id)
                        if rpath.exists():
                            # Only skip if the run was successful (not an error)
                            try:
                                existing = json.loads(rpath.read_text())
                                if existing.get("error") is None and existing.get("metrics") is not None:
                                    skipped += 1
                                    done += 1
                                    continue
                                # else: re-run failed/errored runs
                            except Exception:
                                pass  # corrupt file, re-run

                        label = f"[{done+1}/{total}] {model} | {dataset} | {prompt_type} | run {run_id}"
                        print(f"  {label} ...", end=" ", flush=True)

                        record = {
                            "model": model,
                            "dataset": dataset,
                            "prompt_type": prompt_type,
                            "run_id": run_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "code": None,
                            "raw_response": None,
                            "stdout": None,
                            "stderr": None,
                            "metrics": None,
                            "error": None,
                        }

                        try:
                            # API call
                            raw_response, latency, cost = call_model(model, prompt)
                            record["raw_response"] = raw_response
                            code = extract_code(raw_response)
                            record["code"] = code

                            # Execute
                            success, png_bytes, stdout, stderr, fix_applied = run_code_isolated(
                                code, self.data_dir
                            )
                            record["stdout"] = stdout
                            record["stderr"] = stderr
                            record["fix_applied"] = fix_applied

                            # Save image if success
                            img_path = self.image_path(model, dataset, prompt_type, run_id)
                            if success and png_bytes:
                                img_path.write_bytes(png_bytes)
                                output_path = str(img_path)
                            else:
                                output_path = str(img_path)  # doesn't exist, metrics handles it

                            # Metrics
                            metrics = calculate_all_metrics(
                                code=code,
                                output_path=output_path,
                                dataset_columns=columns,
                                latency=latency,
                                cost=cost,
                            )
                            record["metrics"] = metrics

                            status = "✓" if success else "✗"
                            print(f"{status} ({latency:.1f}s)")

                        except Exception as e:
                            record["error"] = str(e)
                            print(f"✗ ERROR: {e}")

                        # Save result
                        rpath.write_text(json.dumps(record, indent=2, default=str))
                        done += 1

        print(f"\n{'='*70}")
        print(f"Pipeline complete. {done - skipped} new runs, {skipped} skipped (already done).")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*70}\n")
