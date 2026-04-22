#!/usr/bin/env python3
"""
Testing out the inference speed (TPS) from vLLM.
"""

import time
from vllm import LLM, SamplingParams


def main(model_name: str, prompt: str):

    print("Initializing vLLM engine")
    llm = LLM(model=model_name, max_model_len=128, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50) 

    print("vLLM engine ready.")
    print("\nGenerating with vLLM.")
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    generated_text = outputs[0].outputs[0].text
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    # vLLM RESULTS
    print("\nvLLM RESULTS")
    print(f"Generated text: {generated_text[:200]}")
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")

    # Clean up
    del llm


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Once upon a time while learning about LLM inference engines, "
    main(model_name, prompt)
