"""
prompts_v2.py - Rich and poor prompt templates for each dataset.
"""

import pandas as pd
from datasets import load_dataset


def get_rich_prompt(dataset_name: str) -> str:
    """
    Rich prompt: full schema, sample rows, description, explicit task.
    """
    df, meta = load_dataset(dataset_name)
    sample = df.head(3).to_string(index=False)
    col_info = "\n".join([f"  - {col} ({dtype})" for col, dtype in meta["dtypes"].items()])

    prompt = f"""Dataset: {meta['name']}
Description: {meta['description']}
Shape: {meta['shape'][0]} rows × {meta['shape'][1]} columns

Columns:
{col_info}

Sample rows (first 3):
{sample}

Task: Create an insightful, well-labeled visualization of this dataset using matplotlib or seaborn.
- Use appropriate chart type for the data
- Include axis labels, title, and legend where applicable
- Save the final figure as output.png using plt.savefig('output.png', bbox_inches='tight', dpi=150)
- Do not use plt.show()
- Load the data from: data/{dataset_name}.csv (use the exact column names shown above)
"""
    return prompt


def get_poor_prompt(dataset_name: str) -> str:
    """
    Poor prompt: minimal context, just raw CSV snippet, no schema info.
    """
    df, meta = load_dataset(dataset_name)
    csv_snippet = df.head(5).to_csv(index=False)

    prompt = f"""Visualize this dataset and save the result as output.png.

{csv_snippet}
"""
    return prompt


def get_prompt(dataset_name: str, prompt_type: str) -> str:
    """Get prompt by dataset name and type ('rich' or 'poor')."""
    if prompt_type == "rich":
        return get_rich_prompt(dataset_name)
    elif prompt_type == "poor":
        return get_poor_prompt(dataset_name)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}. Use 'rich' or 'poor'.")
