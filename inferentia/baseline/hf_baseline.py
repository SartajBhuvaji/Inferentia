#!/usr/bin/env python3
"""
Testing out the inference speed (TPS) from HuggingFace transformers.
"""

import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_name: str, prompt: str):

    print(f"\nModel: {model_name}")
    print("\nLoading model with HuggingFace transformers...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nModel loaded successfully.")
    print("\nGenerating with HuggingFace transformers...")
    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )
    end_time = time.time()

    # Calculate metrics
    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nRESULTS:")
    print(f"Generated text: {output_text[:200]}...")
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")

    # Clean up
    del model
    del tokenizer


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Once upon a time while learning about LLM inference engines, "
    main(model_name, prompt)
