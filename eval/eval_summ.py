# coding=utf-8
# Copyright 2025 The Google Research Authors.
# Licensed under the Apache License, Version 2.0

"""Evaluate summarization model with LLM-as-judge win rate."""
import click
import random
from datasets import load_dataset
import transformers
from vllm import LLM, SamplingParams
from openai import OpenAI
import gc, torch

from tqdm import tqdm

AutoTokenizer = transformers.AutoTokenizer


# ---------- Judge ----------
client = OpenAI()

def build_judge_prompt(dialogue, summ_a, summ_b):
    return f"""
You are a strict evaluator for dialogue summarization.

Dialogue:
{dialogue}

Summary A:
{summ_a}

Summary B:
{summ_b}

Evaluation criteria (in order of importance):
1. Faithfulness (no hallucinations)
2. Coverage of key information
3. Clarity and conciseness

Which summary is better?

Reply with exactly one token: A, B, or TIE.
"""


def judge(dialogue, a, b, model="gpt-4o-mini"):
    prompt = build_judge_prompt(dialogue, a, b)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def batched_generate(llm, prompts, sampling_params, batch_size=8):
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outs = llm.generate(batch, sampling_params)
        outputs.extend(outs)
    return outputs
# ---------- Main ----------
@click.command()
@click.option('-max_tokens', type=int, default=128)
@click.option('-visualize_text', type=bool, default=False)
@click.option(
    '-ckpt',
    type=str,
    default='summ_1k_seed_20_supervised_kd_kl_350_google-gemma-2b-it',
)
@click.option(
    '-base_model',
    type=str,
    default='Qwen/Qwen2-7B-Instruct',
)
@click.option(
    '-judge_model',
    type=str,
    default='gpt-4o-mini',
)
def main(max_tokens, visualize_text, ckpt, base_model, judge_model):

    # load dataset
    vali_dataset = load_dataset('knkarthick/dialogsum')['test']

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    vali_data = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': d}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for d in vali_dataset['dialogue']
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
    )

    # ---------- Generate summaries ----------
    # llm_A = LLM(model=base_model, tokenizer=base_model)
    # llm_B = LLM(model=ckpt, tokenizer=base_model)

    # outs_A = batched_generate(llm_A, vali_data, sampling_params, batch_size=8)
    # outs_B = batched_generate(llm_B, vali_data, sampling_params, batch_size=8)

    llm_A = LLM(model=base_model, tokenizer=base_model, gpu_memory_utilization=0.7)
    outs_A = batched_generate(llm_A, vali_data, sampling_params, batch_size=8)
    del llm_A
    gc.collect()
    torch.cuda.empty_cache()

    # load only B and generate
    llm_B = LLM(model=ckpt, tokenizer=base_model, gpu_memory_utilization=0.7)
    outs_B = batched_generate(llm_B, vali_data, sampling_params, batch_size=8)
    # ---------- Win-rate evaluation ----------
    wins = {"A": 0, "B": 0, "TIE": 0}

    for dialogue, outA, outB in tqdm(zip(vali_dataset["dialogue"], outs_A, outs_B),
                                    total=len(vali_dataset["dialogue"]),
                                    desc="Judging"):
        summA = outA.outputs[0].text.strip()
        summB = outB.outputs[0].text.strip()

        # order randomization (critical)
        if random.random() < 0.5:
            res = judge(dialogue, summA, summB, judge_model)
            wins[res] += 1
        else:
            res = judge(dialogue, summB, summA, judge_model)
            if res == "A":
                wins["B"] += 1
            elif res == "B":
                wins["A"] += 1
            else:
                wins["TIE"] += 1

        if visualize_text:
            print("Dialogue:", dialogue)
            print("Summary A (base):", summA)
            print("Summary B (distill):", summB)
            print("Judge:", res)
            print(">" * 80)

    total = sum(wins.values())
    win_rate_B = (wins["B"] + 0.5 * wins["TIE"]) / total

    print("===== LLM-as-Judge Results =====")
    print("Wins:", wins)
    print(f"Distilled model win rate: {win_rate_B:.4f}")


if __name__ == '__main__':
    main()