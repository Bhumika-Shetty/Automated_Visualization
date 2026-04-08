"""
api_adapters.py - Stateless API adapters for OpenAI, Anthropic, Google
Every call creates a fresh client with no history.
"""

import os
import time
from typing import Tuple

SYSTEM_PROMPT = (
    "You are an expert data visualization specialist. "
    "Generate Python code using matplotlib/seaborn to create a clear, insightful visualization. "
    "Return ONLY the Python code block, no explanation, no markdown text outside the code block."
)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def call_openai(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call OpenAI API. Returns (response_text, latency_seconds, estimated_cost).
    """
    import openai

    cost_per_1k_input = {"gpt-4o": 0.0025, "gpt-4o-mini": 0.00015}
    cost_per_1k_output = {"gpt-4o": 0.010, "gpt-4o-mini": 0.0006}

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    key = model if model in cost_per_1k_input else "gpt-4o-mini"
    cost = (input_tokens / 1000) * cost_per_1k_input[key] + (output_tokens / 1000) * cost_per_1k_output[key]

    return text, latency, cost


def call_anthropic(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call Anthropic API. Returns (response_text, latency_seconds, estimated_cost).
    """
    import anthropic

    cost_per_1k_input = {"claude-opus-4-5": 0.015, "claude-sonnet-4-5": 0.003}
    cost_per_1k_output = {"claude-opus-4-5": 0.075, "claude-sonnet-4-5": 0.015}

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency = time.time() - start

    text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    key = model if model in cost_per_1k_input else "claude-sonnet-4-5"
    cost = (input_tokens / 1000) * cost_per_1k_input[key] + (output_tokens / 1000) * cost_per_1k_output[key]

    return text, latency, cost


def call_google(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call Google Gemini API via google.genai. Returns (response_text, latency_seconds, estimated_cost).
    """
    import google.genai as genai
    from google.genai import types

    cost_per_1k_input = {
        "models/gemini-2.5-flash": 0.000075,
        "models/gemini-2.5-pro": 0.00125,
        "models/gemini-2.0-flash": 0.0001,
    }
    cost_per_1k_output = {
        "models/gemini-2.5-flash": 0.0003,
        "models/gemini-2.5-pro": 0.010,
        "models/gemini-2.0-flash": 0.0004,
    }

    # Normalize model name — add 'models/' prefix if missing
    if not model.startswith("models/"):
        model = f"models/{model}"

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )
    latency = time.time() - start

    text = response.text
    input_tokens = _estimate_tokens(user_prompt)
    output_tokens = _estimate_tokens(text)
    key = model if model in cost_per_1k_input else "models/gemini-2.5-flash"
    cost = (input_tokens / 1000) * cost_per_1k_input[key] + (output_tokens / 1000) * cost_per_1k_output[key]

    return text, latency, cost


def call_groq(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call Groq API (free tier). Supports Llama, Qwen, DeepSeek, Gemma.
    Returns (response_text, latency_seconds, cost=0).
    """
    import openai

    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    return text, latency, 0.0  # Free tier


def call_deepseek(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call DeepSeek API. Very cheap ($0.14/1M tokens input).
    Returns (response_text, latency_seconds, estimated_cost).
    """
    import openai

    cost_per_1k_input = {"deepseek-chat": 0.00014, "deepseek-coder": 0.00014}
    cost_per_1k_output = {"deepseek-chat": 0.00028, "deepseek-coder": 0.00028}

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    key = model if model in cost_per_1k_input else "deepseek-chat"
    cost = (input_tokens / 1000) * cost_per_1k_input[key] + (output_tokens / 1000) * cost_per_1k_output[key]
    return text, latency, cost


def call_ollama(model: str, user_prompt: str) -> Tuple[str, float, float]:
    """
    Call local Ollama API. Returns (response_text, latency_seconds, cost=0).
    """
    import requests

    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 2048,  # Limit context to reduce memory pressure
            },
        },
        timeout=600,
    )
    latency = time.time() - start

    response.raise_for_status()
    text = response.json()["message"]["content"]
    return text, latency, 0.0  # Local model = $0


def call_model(model: str, user_prompt: str, max_retries: int = 3) -> Tuple[str, float, float]:
    """Route to correct provider based on model name. Retries on 503/rate-limit errors."""
    for attempt in range(max_retries):
        try:
            if model.startswith("gpt"):
                return call_openai(model, user_prompt)
            elif model.startswith("claude"):
                return call_anthropic(model, user_prompt)
            elif model.startswith("gemini"):
                return call_google(model, user_prompt)
            elif model.startswith("deepseek"):
                return call_deepseek(model, user_prompt)
            elif any(model.startswith(p) for p in ["llama", "qwen", "mixtral", "gemma", "qwen/"]):
                return call_groq(model, user_prompt)
            elif ":" in model:  # Ollama format e.g. qwen2.5-coder:32b
                return call_ollama(model, user_prompt)
            else:
                raise ValueError(f"Unknown model: {model}")
        except Exception as e:
            err = str(e)
            retryable = (
                "503" in err or
                "500" in err or
                "UNAVAILABLE" in err or
                "overloaded" in err.lower() or
                "rate" in err.lower() or
                "internal server error" in err.lower()
            )
            if retryable and attempt < max_retries - 1:
                wait = (attempt + 1) * 15  # 15s, 30s, 45s
                import sys
                print(f" [retry {attempt+1}/{max_retries} in {wait}s]", end="", flush=True, file=sys.stderr)
                time.sleep(wait)
                continue
            raise
