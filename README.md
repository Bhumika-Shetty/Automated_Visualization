# Automated Visualization Evaluation Pipeline

**Research project:** "Evaluating the Performance of AI Models in Data Visualization"  
Co-authored with NYU Courant professor.

---

## What This Does

Runs a fully automated benchmark of 6 LLMs on data visualization tasks across 5 datasets, 2 prompt types (rich/poor), and 30 runs per combination — **3,600 total runs**. Measures 11 metrics per run and visualizes results via a Streamlit dashboard.

### Models
| Provider | Models |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `claude-opus-4-5`, `claude-sonnet-4-5` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash` |

### Datasets
`titanic`, `iris`, `sleep_health`, `customer_segments`, `credit_fraud`

### Metrics (11)
`execution_success`, `fidelity_score`, `hallucination_rate`, `color_delta_e`, `visual_entropy`, `labeling_completeness`, `code_complexity`, `generation_latency`, `cost_per_run`, `cross_run_cv`, `prompt_sensitivity_index`

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Bhumika-Shetty/Automated_Visualization.git
cd Automated_Visualization
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a `.env` file in the project root (never committed to git):
```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

---

## Running the Pipeline

### Full run (3,600 runs — production)
```bash
./venv/bin/python3 run_pipeline.py
```

### Test run (quick sanity check)
Edit `run_pipeline.py` temporarily:
```python
p = Pipeline(
    models=["gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash"],
    datasets=["titanic"],
    num_runs=2
)
```

The pipeline is **resumable** — if interrupted, re-running it will skip completed runs automatically.

---

## Viewing Results

### Streamlit Dashboard (local)
```bash
./venv/bin/python3 run_dashboard.py
```
Open: [http://localhost:8501](http://localhost:8501)

### Access from another machine (same network)
```bash
./venv/bin/python3 -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```
Then open: `http://<this-machine-ip>:8501`  
Find your IP with: `ifconfig | grep "inet "` (macOS/Linux)

### Analysis CSVs
```bash
./venv/bin/python3 analysis.py
```
Outputs to `outputs/`:
- `summary_table.csv` — per-model stats
- `prompt_sensitivity.csv` — rich vs poor breakdown
- `provider_summary.csv` — provider-level aggregates

---

## Output Structure

```
outputs/
├── results/       # JSON per run: code, response, metrics, errors
├── images/        # PNG visualizations generated per run
├── summary_table.csv
├── prompt_sensitivity.csv
└── provider_summary.csv
```

---

## Key Design Decisions

- **Stateless runs:** Every API call is a fresh client with zero history — `[system, user]` messages only.
- **Resumable:** Completed runs are skipped on re-run.
- **Resilient execution:** If code crashes due to minor issues (e.g., column name case mismatch), the fix is auto-applied, documented in `fix_applied` field, and the run is retried.
- **Partial success:** If `output.png` is produced even if the script later crashes, it counts.
- **Consistent prompts:** Every model sees the exact same prompt for a given dataset + prompt type combination.

---

## Prompt Types

| Type | Description |
|---|---|
| **Rich** | Full schema with column names, dtypes, 3 sample rows, description, explicit task |
| **Poor** | Just the first 5 rows as raw CSV, minimal instruction |
