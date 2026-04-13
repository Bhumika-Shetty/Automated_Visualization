# Evaluating the Performance of Large Language Models in Automated Data Visualization

**Bhumika Dinesh Shetty¹, [Professor Name]²**

¹ New York University Tandon School of Engineering, Brooklyn, NY  
² New York University Courant Institute of Mathematical Sciences, New York, NY  
`bds9746@nyu.edu`

---

## Abstract

We present a large-scale empirical benchmark evaluating the capability of nine state-of-the-art large language models (LLMs) to perform automated data visualization from tabular datasets. Our benchmark spans **4,319 experimental runs** across 8 models, 8 datasets (5 well-known, 3 unconventional), 2 prompt complexity levels (rich/poor), and 30 independent runs per configuration. We introduce 11 quantitative evaluation metrics covering execution success, data fidelity, hallucination, visual quality, code complexity, and cost-efficiency. Our key findings reveal: (1) GPT-4o-mini achieves the highest execution success rate (97.3%) through conservative code generation; (2) Claude Sonnet 4-5 exhibits a dramatic prompt-sensitivity reversal (96.2% rich vs. 33.3% poor); (3) DeepSeek Chat achieves 90.4% success at near-zero cost ($0.00043/run); (4) open-source models via Groq (Qwen3-32B, Llama 3.1 8B) lag significantly behind closed-source counterparts (41.5%, 66.9%); and (5) models show no systematic advantage on well-known datasets over unconventional ones, challenging the assumption that training data memorization aids visualization performance.

---

## 1. Introduction

The emergence of large language models capable of generating executable code has opened new possibilities for automated data visualization — the task of converting raw tabular data into meaningful visual representations. While LLMs have demonstrated impressive coding capabilities across general programming benchmarks, their reliability for visualization-specific tasks remains poorly understood.

Data visualization presents unique challenges beyond general code generation: models must simultaneously (1) understand dataset schema, (2) select appropriate visualization types, (3) generate syntactically and semantically correct Python code, (4) produce well-labeled, informative charts, and (5) avoid hallucinating non-existent data columns or relationships.

Prior work has evaluated LLMs on coding benchmarks (HumanEval, MBPP), but no systematic study has examined visualization-specific capabilities across diverse datasets, prompt conditions, and model families at scale. This paper addresses that gap with the following contributions:

1. **A reproducible benchmark framework** with stateless, history-free evaluation ensuring each run is a completely independent model query.
2. **4,319 runs** across 9 model families (OpenAI, Anthropic, Google, DeepSeek, open-source via Groq).
3. **11 automated quantitative metrics** covering all aspects of visualization quality.
4. **Known vs. unconventional dataset comparison** testing whether training data memorization affects performance.
5. **Rich vs. poor prompt analysis** revealing dramatic model-specific sensitivity patterns.

---

## 2. Related Work

**Code generation benchmarks.** HumanEval [Chen et al., 2021] and MBPP [Austin et al., 2021] evaluate LLM code generation on algorithmic tasks but do not address visualization or domain-specific data handling. APPS [Hendrycks et al., 2021] evaluates competition programming but lacks data analysis tasks.

**Visualization evaluation.** VizGPT [Huang et al., 2023] and ChartLlama [Han et al., 2023] focus on chart understanding and question answering rather than generation. NL4DV [Narechania et al., 2020] evaluates natural language to visualization translation but predates modern LLMs.

**LLM evaluation frameworks.** HELM [Liang et al., 2022] provides multi-metric evaluation but does not include data visualization as a task category.

To the best of our knowledge, this is the first benchmark to evaluate LLM-generated data visualization at scale using fully automated quantitative metrics.

---

## 3. Methodology

### 3.1 Experimental Setup

**Stateless evaluation protocol.** Every run is completely independent: a fresh API client is instantiated per call, the message history contains exactly two messages [system, user], and code execution occurs in an isolated subprocess within a temporary directory that is destroyed after each run. This ensures zero cross-contamination between runs.

**Models evaluated.** We evaluate 9 models across 5 providers:

| Model | Provider | Type | Approx. Cost/Run |
|---|---|---|---|
| GPT-4o | OpenAI | Closed | $0.0043 |
| GPT-4o-mini | OpenAI | Closed | $0.00024 |
| Claude Opus 4-5 | Anthropic | Closed | $0.100 |
| Claude Sonnet 4-5 | Anthropic | Closed | $0.022 |
| Gemini 2.5 Pro | Google | Closed | $0.0079 |
| Gemini 2.5 Flash | Google | Closed | $0.00017 |
| DeepSeek Chat | DeepSeek | Closed | $0.00043 |
| Qwen3-32B | Groq | Open-Source | $0.00 |
| Llama 3.1 8B | Groq | Open-Source | $0.00 |

**System prompt.** Identical across all models:
> *"You are an expert data visualization specialist. Generate Python code using matplotlib/seaborn to create a clear, insightful visualization. Return ONLY the Python code block, no explanation, no markdown text outside the code block."*

### 3.2 Datasets

We evaluate on 8 tabular datasets divided into two categories:

**Known datasets** (likely present in LLM training corpora):

| Dataset | Rows | Columns | Domain |
|---|---|---|---|
| Titanic | 891 | 15 | Survival prediction |
| Iris | 150 | 5 | Species classification |
| Sleep Health | 374 | 13 | Health analytics |
| Customer Segments | 2,240 | 29 | Marketing |
| Credit Fraud | 1,000 | 6 | Fraud detection |

**Unconventional datasets** (niche, unlikely in training data):

| Dataset | Rows | Columns | Domain |
|---|---|---|---|
| Dogs of Zurich | 103,618 | 35 | Municipal pet registry |
| Cameras | 1,000 | 10 | Consumer electronics |
| Coffee Sales | 2,000 | 10 | Retail transactions |

### 3.3 Prompt Conditions

Each model-dataset combination is evaluated under two prompt types:

**Rich prompt:** Includes full dataset name, all column names with data types, 3 sample rows, dataset description, and explicit visualization instructions including output path specification.

**Poor prompt:** Contains only the first 5 rows as raw CSV text with minimal instruction: *"Visualize this dataset and save the result as output.png."*

### 3.4 Evaluation Metrics

We define 11 automated metrics:

**Execution:**
- **Execution Success (ES):** Binary; whether the generated code executes and produces a valid PNG output. Partial successes (image produced despite later errors) are counted.
- **Generation Latency (GL):** Wall-clock time from API request to response (seconds).

**Data Integrity:**
- **Fidelity Score (FS):** Percentage of actual dataset column names referenced in generated code. Higher = more comprehensive use of available data.
- **Hallucination Rate (HR):** Ratio of quoted string literals in generated code that do not correspond to real column names. Higher = more invented references.

**Visual Quality (computed on successful runs only):**
- **Color ΔE (CDE):** Mean perceptual color difference between adjacent pixels in LAB color space. Higher values indicate greater visual variety.
- **Visual Entropy (VE):** Shannon entropy of pixel value distribution. Higher = more complex, information-dense image.

**Code Quality:**
- **Labeling Completeness (LC):** Ratio of labeling elements present (xlabel, ylabel, title, legend) out of 4. Higher = better documented visualizations.
- **Code Complexity (CC):** Cyclomatic complexity proxy — count of control flow keywords (if, for, while, try). Correlates with visualization ambition.

**Economics:**
- **Cost per Run (CPR):** Estimated cost based on input/output token counts at provider list prices.

**Cross-run (computed across 30 runs):**
- **Cross-Run CV (CRCV):** Coefficient of variation of fidelity score across 30 runs. Lower = more consistent behavior.
- **Prompt Sensitivity Index (PSI):** |Rich% − Poor%| in execution success. Higher = more sensitive to prompt quality.

### 3.5 Auto-Fix Protocol

When code execution fails with column-not-found errors (KeyError, "not in index"), our pipeline applies an automatic fix: it reads actual dataset columns, performs case-insensitive replacement of mismatched column references, and re-executes. The original failure, fix description, and re-execution result are all recorded. Across 4,319 runs, **60 runs (1.4%) triggered the auto-fix mechanism**, confirming this is a minor but real failure mode.

---

## 4. Results

### 4.1 Overall Execution Success

Table 1 presents the complete performance scorecard across all 9 models.

**Table 1: Overall Model Performance Scorecard** *(ranked by execution success)*

| Rank | Model | N | ES% | FS | HR% | LC% | Latency | Cost/Run | CC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-4o-mini | 480 | **97.3** | 58.7 | 54.2 | 85.8 | 6.2s | $0.00024 | 0.70 |
| 2 | Claude Opus 4-5 | 480 | 96.9 | **84.2** | 60.7 | **96.7** | 13.5s | $0.100 | 5.27 |
| 3 | GPT-4o | 480 | 95.4 | 62.0 | 53.8 | 77.7 | **3.7s** | $0.0043 | 0.50 |
| 4 | Gemini 2.5 Flash | 480 | 94.6 | 69.3 | 52.9 | 82.7 | 15.2s | **$0.00017** | 6.23 |
| 5 | Gemini 2.5 Pro | 479 | 90.8 | 73.5 | 65.0 | 87.2 | 30.7s | $0.0079 | 6.85 |
| 6 | DeepSeek Chat | 480 | 90.4 | 84.3 | 59.9 | 91.5 | 26.8s | $0.00043 | 4.30 |
| 7 | Llama 3.1 8B | 480 | 66.9 | 72.4 | **45.1** | 74.5 | 3.9s | $0.00 | 0.82 |
| 8 | Claude Sonnet 4-5 | 480 | 64.8 | 76.7 | 62.1 | 92.6 | 16.2s | $0.022 | 3.75 |
| 9 | Qwen3-32B | 480 | 41.5 | 64.2 | 49.4 | 63.7 | 21.3s | $0.00 | 16.55 |

**Key observation:** GPT-4o-mini achieves the highest execution success (97.3%) despite lower fidelity (58.7) — it generates simpler, more robust code. Claude Opus achieves highest fidelity (84.2) and labeling completeness (96.7%) by writing more comprehensive multi-subplot dashboards, at the cost of slightly higher failure rate. Llama 3.1 8B achieves the lowest hallucination rate (45.1%) among all models.

### 4.2 Failure Analysis

**Table 2: Failure Cause Breakdown**

| Model | Total Fails | Timeout | KeyError | TypeError | ValueError | AttributeError | Other |
|---|---|---|---|---|---|---|---|
| GPT-4o-mini | 13 | 7 | 0 | 0 | 1 | 3 | 2 |
| Claude Opus | 15 | 0 | 1 | 3 | 8 | 0 | 3 |
| GPT-4o | 22 | 10 | 0 | 5 | 4 | 3 | 0 |
| Gemini 2.5 Flash | 26 | 0 | 0 | 1 | 23 | 1 | 1 |
| Gemini 2.5 Pro | 44 | 0 | 1 | 5 | 18 | 17 | 3 |
| DeepSeek Chat | 46 | 0 | 1 | 1 | 19 | 0 | 25 |
| Llama 3.1 8B | 159 | 58 | 4 | 12 | 67 | 13 | 5 |
| Claude Sonnet | 169 | 0 | 0 | 1 | 4 | 20 | 144 |
| Qwen3-32B | 281 | 3 | 0 | 0 | 0 | 15 | 263 |

**Failure pattern analysis:**
- **GPT-4o-mini/GPT-4o:** Primarily timeout failures — models generate complex visualization code that exceeds the 60s execution limit.
- **Gemini:** Dominated by ValueError — models generate calls to matplotlib/seaborn with incorrect parameter types.
- **Llama 3.1 8B:** 58 timeouts (36.5% of failures) suggesting inefficient code generation; 67 ValueErrors.
- **Claude Sonnet:** 144 "other" failures — inspection reveals the model frequently generates markdown prose mixed with code or multi-file outputs that cannot be executed as single scripts.
- **Qwen3-32B:** 263 "other" failures — the model often generates explanatory text, pseudo-code, or refuses the visualization task entirely. Code complexity CV (0.555) is highest of all models, reflecting highly inconsistent behavior.

### 4.3 Code Complexity vs. Success

A notable finding is the relationship between code complexity and execution outcome:

**Table 3: Mean Code Complexity — Successful vs. Failed Runs**

| Model | Success CC | Failure CC | Interpretation |
|---|---|---|---|
| Claude Opus | 5.24 | 6.07 | Complex code fails more |
| Claude Sonnet | 4.36 | 2.64 | Simple code fails more (non-code outputs) |
| DeepSeek Chat | 3.90 | 8.07 | High complexity → failure |
| Gemini 2.5 Flash | 6.11 | 8.27 | High complexity → failure |
| Gemini 2.5 Pro | 6.67 | 8.61 | High complexity → failure |
| Qwen3-32B | 0.07 | **28.22** | Extreme complexity → catastrophic failure |

For most models, more complex code correlates with failure. Qwen3-32B's failed runs have an average complexity of 28.22 — these are cases where the model generates extraordinarily convoluted code that cannot be executed.

### 4.4 Prompt Sensitivity

**Table 4: Prompt Sensitivity Analysis (Rich vs. Poor)**

| Model | Rich ES% | Poor ES% | Gap | Direction | PSI |
|---|---|---|---|---|---|
| Claude Sonnet 4-5 | **96.2** | 33.3 | +62.9pp | Rich >> Poor | **62.9** |
| Qwen3-32B | 58.3 | 24.6 | +33.8pp | Rich >> Poor | 33.8 |
| Llama 3.1 8B | 78.8 | 55.0 | +23.7pp | Rich >> Poor | 23.7 |
| GPT-4o-mini | 100.0 | 94.6 | +5.4pp | Rich > Poor | 5.4 |
| GPT-4o | 97.5 | 93.3 | +4.2pp | Rich > Poor | 4.2 |
| DeepSeek Chat | 88.8 | 92.1 | -3.3pp | Poor > Rich | 3.3 |
| Claude Opus | 96.2 | 97.5 | -1.2pp | Poor > Rich | 1.2 |
| Gemini 2.5 Pro | 90.4 | 91.2 | -0.8pp | Poor > Rich | 0.8 |
| Gemini 2.5 Flash | 94.6 | 94.6 | 0.0pp | Neutral | 0.0 |

**Critical finding:** Claude Sonnet exhibits a catastrophic prompt-sensitivity reversal (+62.9pp). With rich prompts it achieves 96.2% — matching Claude Opus — but with poor prompts drops to 33.3%. Investigation of failure cases reveals that poor prompts cause Sonnet to generate non-executable mixed text/code responses or attempt to read non-existent file paths. Smaller open-source models (Qwen3-32B, Llama 3.1 8B) also show large PSI values, indicating these models rely heavily on prompt scaffolding.

Rich prompts improve performance for all models except Gemini and Claude Opus/DeepSeek, which perform marginally better with minimal prompts — consistent with their higher baseline capability absorbing additional context without confusion.

**Prompt sensitivity by dataset:**

| Dataset | Rich ES% | Poor ES% | Gap |
|---|---|---|---|
| Sleep Health | 90.7 | 67.4 | +23.3pp |
| Titanic | 93.7 | 77.4 | +16.3pp |
| Coffee Sales | 93.0 | 75.8 | +17.1pp |
| Customer Segments | 79.6 | 62.6 | +17.0pp |
| Dogs of Zurich | 82.2 | 70.7 | +11.5pp |
| Cameras | 91.9 | 79.3 | +12.6pp |
| Credit Fraud | 85.2 | 75.6 | +9.6pp |
| Iris | 95.6 | 92.2 | +3.3pp |

Rich prompts consistently improve performance across all datasets. Iris shows the smallest gap (3.3pp) — its simple 5-column structure is easily inferred from raw data. Sleep Health and complex tabular datasets benefit most from schema context.

### 4.5 Known vs. Unconventional Datasets

**Table 5: Known vs. Unconventional Dataset Performance**

| Model | Known ES% | Unknown ES% | Gap | Interpretation |
|---|---|---|---|---|
| GPT-4o-mini | 96.3 | **98.9** | -2.6pp | Unknown > Known |
| Claude Opus | 96.0 | 98.3 | -2.3pp | Unknown > Known |
| GPT-4o | 95.0 | 96.1 | -1.1pp | Unknown > Known |
| Gemini 2.5 Flash | 95.0 | 93.9 | +1.1pp | Known ≈ Unknown |
| Gemini 2.5 Pro | 89.7 | 92.7 | -3.1pp | Unknown > Known |
| DeepSeek Chat | 93.0 | 86.1 | +6.9pp | Known > Unknown |
| Claude Sonnet | 66.7 | 61.7 | +5.0pp | Known > Unknown |
| Llama 3.1 8B | 62.0 | **75.0** | **-13.0pp** | Unknown >> Known |
| Qwen3-32B | 44.3 | 36.7 | +7.7pp | Known > Unknown |

**Finding:** No consistent advantage exists for well-known datasets. Stronger models (GPT-4o-mini, Claude Opus, GPT-4o, Gemini Pro) perform equally or better on unconventional datasets — challenging the hypothesis that training data memorization provides a performance advantage.

Llama 3.1 8B shows the most surprising reversal: 13pp better on unknown datasets. Investigation reveals it struggles with known datasets that have complex multi-column schemas (Customer Segments: 32%, Sleep Health: 63%) but performs well on simpler unconventional datasets (Cameras: 78%, Coffee Sales: 78%).

### 4.6 Per-Dataset Analysis

**Table 6: Success Rate by Dataset (All Models Combined)**

| Dataset | Type | N | ES% | Fidelity | Halluc% | Difficulty |
|---|---|---|---|---|---|---|
| Iris | Known | 540 | **93.9** | 75.9 | 52.9 | Easy |
| Cameras | Unknown | 540 | 85.6 | 78.7 | 52.4 | Moderate |
| Titanic | Known | 540 | 85.6 | 65.1 | 58.8 | Moderate |
| Coffee Sales | Unknown | 539 | 84.4 | 78.3 | 52.8 | Moderate |
| Credit Fraud | Known | 540 | 80.4 | **94.2** | 65.0 | Moderate |
| Sleep Health | Known | 540 | 79.1 | 74.8 | 50.1 | Hard |
| Dogs of Zurich | Unknown | 540 | 76.5 | 42.1 | 60.9 | Hard |
| Customer Segments | Known | 540 | 71.1 | 64.7 | 54.4 | **Hardest** |

Credit Fraud achieves the highest fidelity (94.2) despite moderate success — models correctly reference its 6 simple columns. Dogs of Zurich has the lowest fidelity (42.1) reflecting difficulty handling its 35-column schema. Customer Segments is the hardest dataset overall (71.1% success, complex multi-group structure).

### 4.7 Visual Quality Analysis

**Table 7: Visual Quality Metrics (Successful Runs Only)**

| Model | N_success | Color ΔE | ΔE Std | Visual Entropy | Entropy Std | Labeling% |
|---|---|---|---|---|---|---|
| Claude Opus | 465 | **1.330** | 0.356 | 1.603 | 0.471 | **96.7** |
| Claude Sonnet | 311 | 1.288 | 0.522 | 1.598 | 0.537 | 95.3 |
| Llama 3.1 8B | 321 | 1.190 | 1.236 | 1.118 | 0.600 | 77.6 |
| Qwen3-32B | 199 | 1.212 | 0.645 | 1.334 | 0.775 | 62.2 |
| GPT-4o | 458 | 1.068 | 0.871 | 1.286 | 0.547 | 77.8 |
| GPT-4o-mini | 467 | 1.033 | 1.097 | 1.238 | 0.603 | 85.6 |
| Gemini 2.5 Flash | 454 | 1.050 | 0.456 | 1.217 | 0.655 | 83.2 |
| Gemini 2.5 Pro | 435 | 0.776 | 0.575 | 1.451 | 0.971 | 86.7 |
| DeepSeek Chat | 434 | 0.811 | 0.468 | **1.822** | 0.586 | 91.7 |

**Visual quality findings:**
- Claude Opus produces the highest-quality visualizations: best Color ΔE (1.330), high entropy, and near-perfect labeling (96.7%). Human inspection confirms it consistently generates 2×2 or 3×2 subplot dashboards with professional styling.
- DeepSeek Chat generates the highest visual entropy (1.822) — its visualizations are the most visually complex and information-dense, though at lower color ΔE.
- Gemini Pro has the lowest Color ΔE (0.776) — tends toward monochromatic or muted palette choices.
- GPT-4o-mini achieves near-perfect execution success with moderate visual quality — a pragmatic trade-off.
- Qwen3-32B has the lowest labeling completeness (62.2%) of successful runs — confirming that even when it does generate images, they are less professionally presented.

**Visual quality image analysis** (based on human evaluation of representative samples):

*Claude Opus (Rich, Titanic):* Generates a 2×2 multi-panel dashboard with survival rate by class/sex (grouped bar), age distribution (KDE overlay), survival by embarkation port (bar with counts), and fare distribution by class (grouped boxplot). Consistent color encoding, value annotations on bars, professional typography. Score: **9/10**.

*GPT-4o (Rich, Titanic):* Single grouped bar chart, clean aesthetics, correct axis ordering, clear legend. Concise but appropriately focused. Score: **9/10**.

*GPT-4o-mini (Rich, Titanic):* Grouped bar chart showing survival by class and sex. Minor issue: non-sequential class ordering (3,1,2). Otherwise clear and readable. Score: **6/10** (ordering flaw).

*Gemini Flash (Rich, Titanic):* Grouped bar chart with error bars, correct sequential ordering (First, Second, Third), teal/green palette, well-labeled. Score: **9/10**.

*DeepSeek Chat (Rich, Titanic):* 6-panel dashboard covering class, age histogram, fare boxplot, embarkation, family size, and age group by gender. Highly comprehensive. Score: **9/10**.

*Llama 3.1 8B (Rich, Titanic):* Single bar chart with error bars showing average attribute values by sex — unusual chart choice that doesn't highlight survival. Functional but less insightful. Score: **8/10**.

*Qwen3-32B (Rich, Titanic):* Grouped bar showing survival rate by class and gender. Non-standard class ordering (Third, First, Second), minor title/legend inconsistency. Score: **7/10**.

### 4.8 Hallucination Analysis

**Table 8: Hallucination Rate by Model and Condition**

| Model | Overall% | Rich% | Poor% | Known% | Unknown% |
|---|---|---|---|---|---|
| Llama 3.1 8B | **45.1** | 48.2 | 42.0 | 44.4 | 46.2 |
| Qwen3-32B | 49.4 | 54.8 | 44.1 | 49.3 | 49.6 |
| Gemini 2.5 Flash | 52.9 | 54.8 | 51.0 | 55.1 | 49.2 |
| GPT-4o | 53.8 | 57.7 | 49.9 | 54.1 | 53.3 |
| GPT-4o-mini | 54.2 | 57.8 | 50.6 | 54.5 | 53.7 |
| DeepSeek Chat | 59.9 | 65.9 | 54.0 | 59.7 | 60.3 |
| Claude Opus | 60.7 | 62.1 | 59.3 | 60.8 | 60.5 |
| Claude Sonnet | 62.1 | 62.7 | 61.6 | 61.2 | 63.7 |
| Gemini 2.5 Pro | **65.0** | 65.1 | 64.9 | 67.0 | 61.7 |

**Hallucination finding:** All models exhibit substantial hallucination rates (45–65%), confirming that LLMs freely invent column names, function parameters, and variable names. Counterintuitively, models with higher hallucination (Claude Opus, DeepSeek) achieve higher execution success — these models hallucinate more string literals in comments, docstrings, and text elements rather than functional code paths. Lower-capability models (Llama, Qwen) with lower hallucination rates fail for other reasons (structural errors, non-code outputs).

### 4.9 Latency Analysis

**Table 9: Latency Percentiles (seconds)**

| Model | p50 | p90 | p99 | Mean | Std |
|---|---|---|---|---|---|
| GPT-4o | **3.3** | 5.9 | 10.2 | **3.7** | 1.8 |
| Llama 3.1 8B | 1.0 | 11.9 | 20.7 | 3.9 | 5.0 |
| GPT-4o-mini | 5.1 | 9.8 | 16.6 | 6.2 | 3.2 |
| Claude Opus | 12.9 | 17.3 | 20.8 | 13.5 | 2.6 |
| Gemini 2.5 Flash | 12.8 | 24.6 | 44.9 | 15.2 | 18.9 |
| Claude Sonnet | 16.4 | 21.9 | 25.4 | 16.2 | 4.3 |
| Qwen3-32B | 22.1 | 29.3 | 37.4 | 21.3 | 6.9 |
| DeepSeek Chat | 26.8 | 38.2 | 42.4 | 26.8 | 8.1 |
| Gemini 2.5 Pro | 29.1 | 41.8 | **61.4** | 30.7 | 10.5 |

GPT-4o is fastest at 3.7s mean latency. Llama 3.1 8B has the lowest median (1.0s) but high variance (std=5.0s) due to timeout failures inflating the tail. Gemini 2.5 Flash shows extreme variance (std=18.9s, p99=44.9s) — significant server-side load variation. Gemini Pro is slowest with p99 of 61.4s.

### 4.10 Cost-Efficiency

**Table 10: Cost-Efficiency Analysis**

| Model | Avg Cost/Run | ES% | Cost/Successful Viz | Cost Index (relative to GPT-4o-mini) |
|---|---|---|---|---|
| Llama 3.1 8B | $0.000 | 66.9 | $0.000 | 0× |
| Qwen3-32B | $0.000 | 41.5 | $0.000 | 0× |
| Gemini 2.5 Flash | $0.000173 | 94.6 | $0.000183 | 0.7× |
| GPT-4o-mini | $0.000242 | 97.3 | $0.000249 | **1.0×** |
| DeepSeek Chat | $0.000431 | 90.4 | $0.000477 | 1.9× |
| GPT-4o | $0.004330 | 95.4 | $0.004538 | 18.2× |
| Gemini 2.5 Pro | $0.007944 | 90.8 | $0.008747 | 35.1× |
| Claude Sonnet | $0.021684 | 64.8 | $0.033468 | 134× |
| Claude Opus | $0.100396 | 96.9 | $0.103634 | **416×** |

**Cost-efficiency finding:** GPT-4o-mini represents the best value among paid models ($0.000249/successful viz), with Gemini 2.5 Flash being marginally cheaper ($0.000183) at slightly lower success rate. Claude Opus is 416× more expensive per successful visualization than GPT-4o-mini while achieving only 0.4pp higher success rate. Claude Sonnet is particularly poor value: 134× GPT-4o-mini's cost for significantly lower success rate (64.8%).

### 4.11 Provider-Level Summary

**Table 11: Provider Aggregates**

| Provider | Models | N | ES% | Fidelity | Halluc% | Mean Latency | Mean Cost |
|---|---|---|---|---|---|---|---|
| OpenAI | GPT-4o, 4o-mini | 960 | **96.4** | 60.4 | 54.0 | 5.0s | $0.00229 |
| DeepSeek | DeepSeek Chat | 480 | 90.4 | **84.3** | 59.9 | 26.8s | $0.00043 |
| Google | Gemini 2.5 Pro/Flash | 959 | 92.7 | 71.4 | 59.0 | 23.0s | $0.00405 |
| Anthropic | Claude Opus/Sonnet | 960 | 80.8 | 80.5 | 61.4 | 14.9s | $0.06104 |
| Groq (OSS) | Qwen3-32B, Llama 3.1 8B | 960 | 54.2 | 68.3 | **47.3** | 12.6s | $0.00000 |

---

## 5. Discussion

### 5.1 The Capability-Ambition Trade-off

Our results reveal a fundamental trade-off between model capability and code ambition. Higher-capability models (Claude Opus, DeepSeek) generate more ambitious multi-panel visualizations with higher code complexity (CC=5.27, 4.30), resulting in better fidelity and visual quality but marginally lower execution success rates. Lower-capability models (GPT-4o-mini, GPT-4o) generate simpler, more conservative code (CC=0.70, 0.50) that executes reliably but produces less comprehensive visualizations.

### 5.2 The Claude Sonnet Anomaly

Claude Sonnet 4-5's extreme prompt sensitivity (PSI=62.9pp) is the most surprising finding in our study. With rich prompts, it performs comparably to Claude Opus (96.2% vs 96.9%). With poor prompts, it catastrophically fails (33.3%). Analysis of failure cases reveals Sonnet frequently generates non-executable prose descriptions of what the code *should* do, or generates multi-block responses that cannot be executed as single Python scripts. This suggests Sonnet is more sensitive to task framing than Opus, which appears more robust to prompt quality variation (PSI=1.2pp).

### 5.3 Training Data Memorization Does Not Explain Performance

The conventional assumption that LLMs perform better on datasets present in their training data is not supported by our results. Most models perform comparably or better on unconventional datasets (Cameras, Coffee Sales) than on well-known benchmarks (Customer Segments, Sleep Health). This finding suggests that visualization success depends more on schema comprehension ability and code generation skill than on dataset-specific memorization.

### 5.4 Open-Source Gap

A substantial performance gap exists between closed-source and open-source models. The best open-source model (Llama 3.1 8B, 66.9%) lags 30.4pp behind the best closed-source model (GPT-4o-mini, 97.3%). Qwen3-32B (41.5%) performs worse than random chance for some dataset-prompt combinations. This gap appears driven by both code generation capability and instruction following — open-source models more frequently generate non-executable outputs.

### 5.5 DeepSeek as a Cost-Efficient Alternative

DeepSeek Chat achieves 90.4% execution success, highest fidelity (84.3), and strong labeling completeness (91.5%) at near-zero cost ($0.00043/run). For applications requiring comprehensive data analysis rather than reliable quick visualization, DeepSeek offers an exceptional value proposition.

---

## 6. Limitations

1. **Code execution environment:** All code is executed in isolated subprocesses with a 60-second timeout. Production visualization tasks may use different environments (Jupyter, web-based tools) with different failure modes.
2. **Metric subjectivity:** Fidelity and hallucination rates are computed via string matching, which may misclassify legitimate variable names as hallucinations.
3. **Dataset generation:** Three of our eight datasets (Cameras, Credit Fraud, Coffee Sales) are synthetically generated, which may not capture the full complexity of real-world data.
4. **Model versioning:** Model APIs evolve continuously; our results reflect specific model versions evaluated between March–April 2026.
5. **Gemini Pro completeness:** One Gemini 2.5 Pro run failed with a 404 API error; excluded from success counts.

---

## 7. Conclusion

We have presented the most comprehensive benchmark of LLM data visualization capability to date, spanning 4,319 runs across 9 models, 8 datasets, and 11 automated metrics. Our key findings are:

1. **GPT-4o-mini achieves the highest execution success (97.3%)** through conservative, robust code generation — not through superior data comprehension.
2. **Claude Opus generates the highest-quality visualizations** (fidelity 84.2, labeling 96.7%) but at 416× the cost of GPT-4o-mini per successful visualization.
3. **Claude Sonnet exhibits a catastrophic prompt-sensitivity reversal** (62.9pp gap), making it unreliable in production settings without careful prompt engineering.
4. **DeepSeek Chat is the best value alternative** (90.4% success, highest fidelity, near-zero cost).
5. **Open-source models lag substantially** (41.5–66.9% success) behind closed-source counterparts (90.4–97.3%).
6. **Training data memorization does not drive performance** — models generalize to unconventional datasets as well as or better than to well-known ones.
7. **All models hallucinate heavily** (45–65%), confirming the need for post-generation validation in any production visualization pipeline.

These findings have direct implications for practitioners deploying LLM-based visualization systems: GPT-4o-mini or Gemini 2.5 Flash offer the best reliability/cost trade-offs; Claude Opus maximizes visual quality when cost is not a constraint; and open-source alternatives remain substantially below production-grade reliability thresholds.

---

## References

[1] Chen, M., et al. (2021). Evaluating large language models trained on code. *arXiv:2107.03374*.

[2] Austin, J., et al. (2021). Program synthesis with large language models. *arXiv:2108.07732*.

[3] Hendrycks, D., et al. (2021). Measuring coding challenge competence with APPS. *NeurIPS 2021*.

[4] Liang, P., et al. (2022). Holistic evaluation of language models. *arXiv:2211.09110*.

[5] Narechania, A., et al. (2020). NL4DV: A toolkit for generating analytic specifications for data visualization from natural language queries. *IEEE TVCG*.

[6] Han, J., et al. (2023). ChartLlama: A multimodal LLM for chart understanding and generation. *arXiv:2311.16483*.

---

## Appendix A: Experimental Setup Details

**API System Prompt (identical for all models):**
```
You are an expert data visualization specialist. Generate Python code using 
matplotlib/seaborn to create a clear, insightful visualization. Return ONLY 
the Python code block, no explanation, no markdown text outside the code block.
```

**Rich Prompt Template:**
```
Dataset: {name}
Description: {description}
Shape: {rows} rows × {columns} columns
Columns: {col: dtype for each column}
Sample rows (first 3): {csv_head}
Task: Create an insightful, well-labeled visualization. Save as output.png using
plt.savefig('output.png', bbox_inches='tight', dpi=150). Do not use plt.show().
Load data from: data/{name}.csv
```

**Poor Prompt Template:**
```
Visualize this dataset and save the result as output.png.
{first 5 rows as raw CSV}
```

**Code execution parameters:** Python 3.x subprocess, 60-second timeout, isolated temporary directory, data files copied per run, output.png path patched to absolute path.

**Auto-fix trigger conditions:** `KeyError`, `column` in stderr, `not in index` in stderr.

## Appendix B: Dataset Details

**Dogs of Zurich:** Municipal dog registration dataset from the City of Zurich Open Data portal. 103,618 records, 35 columns including breed, age, gender, district, owner demographics. Represents a maximally unconventional dataset with domain-specific German column names.

**Cameras:** Synthetic dataset of 1,000 digital camera specifications (brand, model year, megapixels, max ISO, weight, price, sensor type, 4K capability, lens mount, body type). Generated with realistic distributions based on consumer camera market data.

**Coffee Sales:** Synthetic transaction-level dataset (2,000 rows) representing coffee shop sales including product, size, price, payment method, customer demographics, and takeaway status.
