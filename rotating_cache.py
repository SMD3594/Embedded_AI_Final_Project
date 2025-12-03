# rotating_cache.py  (Phase 2.3: Rotating KV cache inference)

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kv_cache_manager import RotatingKVCache

def run_rotating(
    prompt: str = "Hello, how are you?",
    max_new_tokens: int = 50,
    max_tokens_in_cache: int = 128,
):
    print("Loading model...")
    start_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize rotating KV cache
    kv_cache = RotatingKVCache(
        max_tokens=max_tokens_in_cache,
        key_dim=None,
        value_dim=None,
    )

    print(f"Generating text for prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    start_gen = time.time()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        kv_cache.append(past_key_values, past_key_values)

        generated_ids = input_ids

        for step in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            outputs = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Rotating cache will overwrite the oldest entry when full
            kv_cache.append(past_key_values, past_key_values)

        gen_time = time.time() - start_gen

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n=== Generated Output (Rotating KV cache) ===")
    print(decoded)
    print("========================\n")
    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"Avg time per token: {gen_time / max_new_tokens:.4f} s/token")

    print(f"RotatingKVCache --> size: {kv_cache.size}, "
          f"buffer length: {len(kv_cache.cached_K)}")

    return decoded

if __name__ == "__main__":
    run_rotating()
