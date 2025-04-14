"""Benchmark offline inference throughput.
https://github.com/mlcommons/inference/tree/master/language/mixtral-8x7b
"""
import argparse
import dataclasses
import json
import random
import time
from typing import List, Optional, Tuple
 
import torch
import uvloop
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from contextlib import contextmanager, nullcontext
from rpdTracerControl import rpdTracerControl as rpd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.manual_seed(0)

def get_llm_instance():
    return LLM(model="/scratch/bingxche/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=4,  
            max_seq_len_to_capture=3072, gpu_memory_utilization=0.97, block_size=32, max_num_seqs=3840, enable_chunked_prefill = False,
            enforce_eager = True, disable_log_stats = False,
            num_scheduler_steps=9, max_num_batched_tokens=32768)
              
def sample_requests_moe():
    processed_data = pd.read_pickle("/scratch/bingxche/data/09292024_mixtral_15k_mintoken2_v1.pkl").iloc[0:100]
    
    prompts: List[Tuple[int, int, int]] = []
    for idx, request in processed_data.iterrows():
        prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        # prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        # prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
        # prompts.append((request['tok_input'], request['tok_input_len'], request['tok_ref_output_len']))
   
    # sort prompts by descedning length
    sorted_prompts = sorted(prompts, key=lambda x: x[1], reverse=True)
    return sorted_prompts
 
def profile_run_vllm(prompts, sampling_params):
    llm = get_llm_instance()
    profile_dir = "/home/bingxche/trace_dir"
    with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))) as p:
                        start = time.perf_counter()
                        llm.generate(prompt_token_ids=prompts, sampling_params = sampling_params, use_tqdm=True)
                        end = time.perf_counter()
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print("time:", end - start)

def run_vllm(prompts, sampling_params):
    llm = get_llm_instance()
    start = time.perf_counter()
    responses = llm.generate(prompt_token_ids=prompts, sampling_params = sampling_params, use_tqdm=True)
    logger.info(f"first 10 rows of responses: {responses[:10]}")
    end = time.perf_counter()
    output_prompts = [response.outputs[0].token_ids for response in responses]
    print("time:", end - start)
    out = [len(a) for a in output_prompts]
    print("output prompt lens:", sum(out) / len(out))
    print("input and output prompts:", output_prompts[0])
    return (prompts, output_prompts, end - start)
 
def main(enable_profiling: False):
    requests = sample_requests_moe()
    # Add the requests to the engine.
    prompts: List[int] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=1.0,
                top_k = 1,
                top_p=0.001,
                ignore_eos=False,
                max_tokens=1024,
            ))
    
    if enable_profiling:
        profile_run_vllm(prompts, sampling_params)
    else:
        prompts, output_prompts, elapsed_time = run_vllm(prompts, sampling_params)
        assert len(prompts) == len(output_prompts), "prompt input and output lengths are different"
        total_num_tokens = sum(len(prompts[idx]) + len(output_prompts[idx]) for idx in range(0, len(prompts)))
        total_output_tokens = sum(len(output_prompt) for output_prompt in output_prompts)
        print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
            f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
            f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
 
if __name__ == "__main__":
    main(False)
