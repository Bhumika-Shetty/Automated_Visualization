# Results Analysis: 1,800 Experimental Runs (March 30, 2026)

This directory contains the detailed analysis of 1,800 experimental runs evaluating six state-of-the-art large language models on automated data visualization generation.

## Key Findings Summary

### Overall Model Ranking (Weighted Composite)
1.  **Claude Sonnet 4 (Anthropic):** Top-performing model.
    *   Success: 88.7%
    *   Fidelity: 19.3
    *   Labeling: 77.9%
    *   Hallucination: 79.0%
    *   Latency: 16.0s
    *   Cost: $0.02
2.  **Claude Opus 4 (Anthropic):**
    *   Success: 74.7%
    *   Fidelity: 19.3
    *   Labeling: 78.8%
    *   Hallucination: 79.3%
    *   Latency: 26.0s
    *   Cost: $0.11
3.  **GPT-4o (OpenAI):**
    *   Success: 70.7%
    *   Fidelity: 6.4
    *   Labeling: 64.4%
    *   Hallucination: 70.3%
    *   Latency: 4.9s
    *   Cost: $0.00
4.  **GPT-4o-mini (OpenAI):**
    *   Success: 64.0%
    *   Fidelity: 7.3
    *   Labeling: 65.4%
    *   Hallucination: 64.6%
    *   Latency: 12.6s
    *   Cost: $0.00
5.  **Gemini 2.5 Pro (Google):**
    *   Success: 28.3%
    *   Fidelity: 15.2
    *   Labeling: 46.8%
    *   Hallucination: 79.8%
    *   Latency: 28.3s
    *   Cost: $0.01
6.  **Gemini 2.5 Flash (Google):**
    *   Success: 26.0%
    *   Fidelity: 4.6
    *   Labeling: 51.3%
    *   Hallucination: 74.1%
    *   Latency: 12.8s
    *   Cost: $0.00

### Key Insights
*   **Claude Sonnet 4 is the top-performing model overall** (88.7% execution success).
*   **Gemini models show near-zero success on rich prompts** (2.7% and 0.7%), indicating a critical failure mode.
*   **Prompt-sensitivity reversal:** OpenAI models perform better with rich prompts (14-21 points better), while Anthropic and Google models perform significantly better with poor prompts (12-51 points better).
*   **High Hallucination:** Universally high across all models (64–80%), requiring human review.
*   **Claude models lead on data fidelity and labeling quality.**
*   **Claude Opus 4 is Pareto-dominated by Sonnet 4** (5.6x more expensive, 63% slower, lower success rate).

## Contents
*   `results_analysis_1.pdf`: The full supplementary document with detailed analysis, tables, and figures.
*   (Additional structured data will be extracted and added here.)
