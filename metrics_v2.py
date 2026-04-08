"""
metrics_v2.py - 11 evaluation metrics for each pipeline run.
"""

import re
import math
from pathlib import Path
from typing import Optional


def execution_success(output_path: str) -> bool:
    return Path(output_path).exists() and Path(output_path).stat().st_size > 0


def fidelity_score(code: str, dataset_columns: list) -> float:
    """How many real column names appear in the generated code (0-100)."""
    if not dataset_columns:
        return 0.0
    found = sum(1 for col in dataset_columns if col in code)
    return round((found / len(dataset_columns)) * 100, 2)


def hallucination_rate(code: str, dataset_columns: list) -> float:
    """
    Ratio of column-like identifiers in code that are NOT real columns.
    Proxy: extract quoted strings and variable names that look like column names.
    """
    # Find all quoted strings in code (likely column references)
    quoted = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_ ]*)['\"]", code)
    if not quoted:
        return 0.0
    col_set = set(col.lower() for col in dataset_columns)
    hallucinated = sum(1 for q in quoted if q.lower() not in col_set and len(q) > 1)
    return round(hallucinated / len(quoted), 4)


def color_delta_e(image_path: str) -> Optional[float]:
    """Mean perceptual color difference between adjacent pixels (proxy using LAB space)."""
    try:
        from skimage import io, color
        import numpy as np
        img = io.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img_float = img.astype(float) / 255.0
        lab = color.rgb2lab(img_float)
        # Compute difference between adjacent pixels (horizontal)
        diff = lab[:, 1:, :] - lab[:, :-1, :]
        delta_e = np.sqrt((diff ** 2).sum(axis=2))
        return round(float(delta_e.mean()), 4)
    except Exception:
        return None


def visual_entropy(image_path: str) -> Optional[float]:
    """Shannon entropy of pixel values."""
    try:
        from skimage import io
        import numpy as np
        img = io.imread(image_path)
        if img.ndim == 3:
            img = img.mean(axis=2)
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0].astype(float)
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return round(float(entropy), 4)
    except Exception:
        return None


def labeling_completeness(code: str) -> float:
    """
    Ratio of labeling elements present: xlabel, ylabel, title, legend.
    """
    checks = [
        bool(re.search(r"xlabel|set_xlabel", code)),
        bool(re.search(r"ylabel|set_ylabel", code)),
        bool(re.search(r"title|set_title|suptitle", code)),
        bool(re.search(r"legend|label=", code)),
    ]
    return round(sum(checks) / len(checks), 4)


def code_complexity(code: str) -> int:
    """Cyclomatic complexity proxy: count control flow keywords."""
    keywords = ["if ", "elif ", "else:", "for ", "while ", "try:", "except ", "with "]
    return sum(code.count(kw) for kw in keywords)


def prompt_sensitivity_index() -> None:
    """Computed across runs in analysis.py, not per-run."""
    return None


def cross_run_cv() -> None:
    """Computed across runs in analysis.py, not per-run."""
    return None


def calculate_all_metrics(
    code: str,
    output_path: str,
    dataset_columns: list,
    latency: float,
    cost: float,
) -> dict:
    """Calculate all 11 metrics for a single run."""
    success = execution_success(output_path)
    return {
        "execution_success": success,
        "fidelity_score": fidelity_score(code, dataset_columns),
        "hallucination_rate": hallucination_rate(code, dataset_columns),
        "color_delta_e": color_delta_e(output_path) if success else None,
        "visual_entropy": visual_entropy(output_path) if success else None,
        "labeling_completeness": labeling_completeness(code),
        "code_complexity": code_complexity(code),
        "generation_latency": round(latency, 3),
        "cost_per_run": round(cost, 6),
        "cross_run_cv": None,
        "prompt_sensitivity_index": None,
    }
