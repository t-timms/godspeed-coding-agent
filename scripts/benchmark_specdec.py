"""Benchmark llama.cpp speculative decoding with 1.5B draft model."""

from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

# Coding prompt to stress-test token generation
MESSAGES = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {
        "role": "user",
        "content": (
            "Write a Python function that implements merge sort with type hints, "
            "docstring, and unit tests using pytest. Include edge case handling."
        ),
    },
]

PAYLOAD = {
    "model": "qwen2.5-coder-14b",
    "messages": MESSAGES,
    "max_tokens": 512,
    "temperature": 0,
    "top_k": 1,
    "stream": False,
}


def benchmark() -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(PAYLOAD).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # llama.cpp server includes timing stats in usage
    timings = usage.get("timings", {})
    prompt_ms = timings.get("prompt_ms", 0)
    predicted_ms = timings.get("predicted_ms", 0)

    tok_s = 0.0
    if predicted_ms and predicted_ms > 0:
        tok_s = completion_tokens / (predicted_ms / 1000.0)

    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "tok_s": tok_s,
        "text_preview": data["choices"][0]["message"]["content"][:200],
    }


def get_server_props() -> dict:
    req = urllib.request.Request("http://127.0.0.1:8080/props", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    print("Warming up...")
    try:
        benchmark()
    except Exception as exc:
        print("Warm-up failed:", exc)

    print("Running benchmark (3 iterations)...")
    results = []
    for i in range(3):
        try:
            r = benchmark()
            results.append(r)
            print(
                f"Run {i + 1}: {r['tok_s']:.1f} tok/s"
                f" | {r['completion_tokens']} tokens"
                f" | {r['predicted_ms']:.0f} ms"
            )
        except Exception as exc:
            print(f"Run {i + 1} failed:", exc)

    if results:
        avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
        avg_tokens = sum(r["completion_tokens"] for r in results) / len(results)
        print(f"\nAverage: {avg_tok_s:.1f} tok/s | {avg_tokens:.0f} tokens/response")

    # Try to read acceptance rate from server props (if available)
    try:
        props = get_server_props()
        print("\nServer props:")
        print(json.dumps(props, indent=2)[:1000])
    except Exception as exc:
        print("Could not fetch /props:", exc)
