# baseline_inference.py
# Phase 2.2: Quantized Cache for DistilGPT-2
# This script compresses K/V Tensors

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCache

def run_baseline(prompt):
    print("Loading model...")
    start_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    #model = AutoModelForCausalLM.from_pretrained(
    #    "distilgpt2",
    #    dtype=torch.float16,
    #    device_map="auto"
    #)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.\n")

    # Prepare inputs
    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Generating with quantized cache...")
    start_gen = time.time()

    output = model.generate(
        **inputs, 
        do_sample=False, 
        max_new_tokens=50, 
        cache_implementation="quantized", 
        cache_config={"backend": "hqq", "nbits": 8}# Change nbits to adjust Quantize Level (1,2,3,4,8)
    )

    gen_time = time.time() - start_gen

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    #return decoded
    print("\n=== Generated Output ===")
    print(decoded)
    print("========================\n")

    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"Avg time per token: {gen_time / 50:.4f} s/token")

if __name__ == "__main__":
     run_baseline("Hello, my name is")
