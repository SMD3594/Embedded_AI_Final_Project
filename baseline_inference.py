# baseline_inference.py
# Phase 1: Baseline inference for DistilGPT-2
# This script verifies model loading, generation, and timing.

import time
import os
import psutil

from transformers import AutoTokenizer, AutoModelForCausalLM


def run_baseline(prompt="Hello, how are you?"):
    print("Loading model...")
    start_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.\n")

    print(f"Generating text for prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt")  # inputs["input_ids"].shape = (1, seq_len)

    # ----- METRICS: RAM + timing -----
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    start_gen = time.time()

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )

    end_gen = time.time()
    elapsed = end_gen - start_gen

    rss_after = proc.memory_info().rss
    rss_peak = max(rss_before, rss_after)
    rss_peak_mb = rss_peak / (1024 * 1024)

    # how many *new* tokens were generated
    num_new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    avg_latency = elapsed / num_new_tokens
    tokens_per_sec = num_new_tokens / elapsed

    # ----- Decode and print text -----
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n=== Generated Output ===")
    print(decoded)
    print("=======================\n")

    # ----- Print metrics -----
    print("=== METRICS (Raspberry Pi) ===")
    print(f"New tokens generated: {num_new_tokens}")
    print(f"Total generation time: {elapsed:.4f} s")
    print(f"Average latency per token: {avg_latency:.4f} s/token")
    print(f"Tokens per second: {tokens_per_sec:.2f} tokens/s")
    print(f"Peak RAM usage (approx): {rss_peak_mb:.2f} MB")


if __name__ == "__main__":
    run_baseline()

