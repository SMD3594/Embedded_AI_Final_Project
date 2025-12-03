# baseline_inference.py
# Phase 1: Baseline inference for DistilGPT-2
# This script verifies model loading, generation, and timing.

import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_baseline(prompt="Hello, how are you?"):
    print("Loading model...")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.\n")

    print(f"Generating text for prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt")

    start_gen = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )
    gen_time = time.time() - start_gen

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n=== Generated Output ===")
    print(decoded)
    print("========================\n")

    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"Avg time per token: {gen_time / 50:.4f} s/token")

if __name__ == "__main__":
    run_baseline()
