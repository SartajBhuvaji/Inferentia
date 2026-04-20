#!/usr/bin/env python3
"""
Testing out the inference speed (TPS) from HuggingFace transformers.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_name: str, prompt: str):
    print(f"\nModel: {model_name}")
    print("\nLoading model with HuggingFace transformers...")

    # Reset GPU
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Model loaded. Peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
    )
    print("\nGenerating...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Start profiling
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,   # helps identify which ops are taking time
            profile_memory=True,  # tracks GPU memory usage per op
            with_stack=True,      # captures call stack for each op, useful for debugging
            with_flops=True,      # estimates FLOPs for supported ops, gives insight into computational cost
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler_logs"),
        ) as prof
    ):
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()
    #  End of Profiling

    # Calculate metrics
    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    print("\nRESULTS:")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")
    print(
        f"Peak GPU memory during generation: {torch.cuda.max_memory_allocated() / 1024**2:.3f} MB"
    )

    # Show top-10 time-consuming ops
    print("\nTop 10 CUDA operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Once upon a time while learning about LLM inference engines, "
    main(model_name, prompt)
