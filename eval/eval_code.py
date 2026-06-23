
# coding=utf-8
"""
Evaluate a model on HumanEval (execution-based) using openai/human-eval harness.

Changes vs raw-prompt version:
- Wrap each HumanEval prompt using tokenizer.apply_chat_template(...) (instruct-style).
- Remove forced leading newline in completion (can break indentation).
- Prompt-aware postprocess: if raw prompt ends with indentation, strip leading newlines.
- Use minimal stop (only ```), to avoid truncating code.

Usage:
CUDA_VISIBLE_DEVICES=3 python eval/eval_code.py \
  --ckpt <MODEL_PATH> \
  --tokenizer <TOKENIZER_NAME_OR_PATH> \
  --temperature 0.2 \
  --n_samples 20 \
  --out humaneval_samples.jsonl \
  --run_eval True
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import click
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _strip_code_fences(s: str) -> str:
    """Remove ```python ... ``` fences if model emits them."""
    s = s.strip()
    s = re.sub(r"^\s*```(?:python)?\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.rstrip()


def _prompt_ends_with_indent(prompt: str) -> bool:
    """True if prompt ends with spaces/tabs (common in HumanEval)."""
    return re.search(r"[ \t]+$", prompt) is not None


def _postprocess_completion(raw_prompt: str, completion: str) -> str:
    """
    HumanEval expects 'completion' to be appended to the provided *raw prompt*.

    Key:
    - DO NOT force a leading newline.
    - If prompt ends with indentation, strip leading newlines from completion.
    """
    c = _strip_code_fences(completion).rstrip()

    # If raw prompt already ends with indentation, leading newline may break indentation.
    if _prompt_ends_with_indent(raw_prompt):
        c = c.lstrip("\n")

    # Generally safe: remove leading blank lines (newlines only)
    c = c.lstrip("\n")
    return c


def _build_chat_prompt(
    raw_prompt: str,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Wrap HumanEval prompt in an instruct chat template.

    NOTE:
    - Some tokenizers (e.g., Gemma2-it) do NOT support system role.
    - Use user-only message for broad compatibility.
    """
    user_msg = (
        "Complete the following Python function.\n"
        "Return ONLY valid Python code that continues the given prompt.\n"
        "Do not include explanations, markdown, or code fences.\n\n"
        f"{raw_prompt}"
    )
    messages = [{"role": "user", "content": user_msg}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@click.command()
@click.option("--ckpt", type=str, required=True, help="Model checkpoint/path for vLLM.")
@click.option("--tokenizer", type=str, default=None, help="Tokenizer name/path (recommended for chat template).")
@click.option("--max_tokens", type=int, default=768)
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.95)
@click.option("--n_samples", type=int, default=1, help="How many samples per task (for pass@k).")
@click.option("--out", type=str, default="humaneval_samples.jsonl")
@click.option("--run_eval", type=bool, default=True, help="Run human-eval after generation.")
def main(ckpt, tokenizer, max_tokens, temperature, top_p, n_samples, out, run_eval):
    # 1) Load HumanEval problems
    ds = load_dataset("openai/openai_humaneval")  # 164 tasks
    split = "test" if "test" in ds else list(ds.keys())[0]
    problems = ds[split]

    # 2) Load tokenizer for chat template rendering
    tok_name = tokenizer or ckpt
    tok: Optional[AutoTokenizer] = AutoTokenizer.from_pretrained(tok_name)

    # 3) vLLM setup
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n_samples,
        # For code, keep stops minimal to avoid truncation.
        stop=["```"],
    )
    llm = LLM(model=ckpt, tokenizer=tok_name)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 4) Generate + write samples.jsonl
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(problems)):
            task_id = problems[i].get("task_id", f"HumanEval/{i}")
            raw_prompt = problems[i]["prompt"]

            chat_prompt = _build_chat_prompt(raw_prompt, tok)
            outputs = llm.generate([chat_prompt], sampling_params)[0].outputs

            for o in outputs:
                completion = _postprocess_completion(raw_prompt, o.text)
                row = {"task_id": task_id, "completion": completion}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote samples: {out_path.resolve()}")

    # 5) Run execution-based evaluation (official harness)
    if run_eval:
        cmd = ["python", "-m", "human_eval.evaluate_functional_correctness", str(out_path)]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
############################################
########## original version ################
# # coding=utf-8
# """
# Evaluate a model on HumanEval (execution-based) using openai/human-eval harness.

# How it works:
# 1) Load problems (HF dataset: openai/openai_humaneval)
# 2) Generate code completions with vLLM
# 3) Write samples.jsonl lines: {"task_id": "...", "completion": "..."}
# 4) Run human-eval evaluator: python -m human_eval.evaluate_functional_correctness samples.jsonl
# """

# import json
# import re
# import subprocess
# from pathlib import Path

# import click
# from datasets import load_dataset
# from vllm import LLM, SamplingParams


# def _strip_code_fences(s: str) -> str:
#     """Remove ```python ... ``` fences if model emits them."""
#     s = s.strip()
#     # remove leading ```lang
#     s = re.sub(r"^\s*```(?:python)?\s*\n", "", s, flags=re.IGNORECASE)
#     # remove trailing ```
#     s = re.sub(r"\n\s*```\s*$", "", s)
#     return s.rstrip()


# def _postprocess_completion(completion: str) -> str:
#     """
#     HumanEval expects 'completion' to be appended to the provided prompt.
#     Make it safe-ish:
#       - strip code fences
#       - ensure it starts with a newline (common in human-eval examples)
#       - avoid trailing chatty text
#     """
#     c = _strip_code_fences(completion)

#     # If the model repeats the prompt or function signature, it's usually still OK,
#     # but to keep consistent with harness expectation, we keep whatever it generated.
#     # Ensure it begins with newline to append cleanly.
#     if not c.startswith("\n"):
#         c = "\n" + c

#     return c


# @click.command()
# @click.option("--ckpt", type=str, required=True, help="Model checkpoint/path for vLLM.")
# @click.option("--tokenizer", type=str, default=None, help="Tokenizer name/path (optional).")
# @click.option("--max_tokens", type=int, default=512)
# @click.option("--temperature", type=float, default=0.8)
# @click.option("--top_p", type=float, default=0.95)
# @click.option("--n_samples", type=int, default=1, help="How many samples per task (for pass@k).")
# @click.option("--out", type=str, default="humaneval_samples.jsonl")
# @click.option("--run_eval", type=bool, default=True, help="Run human-eval after generation.")
# def main(ckpt, tokenizer, max_tokens, temperature, top_p, n_samples, out, run_eval):
#     # 1) Load HumanEval problems
#     ds = load_dataset("openai/openai_humaneval")  # 164 tasks
#     split = "test" if "test" in ds else list(ds.keys())[0]
#     problems = ds[split]

#     # 2) vLLM setup
#     sampling_params = SamplingParams(
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         n=n_samples,
#         # Common stop signals for instruct models (adjust if needed)
#         stop=["```", "\n\n\n", "\r\n\r\n\r\n"],
#     )
#     llm = LLM(model=ckpt, tokenizer=tokenizer or ckpt)

#     out_path = Path(out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # 3) Generate + write samples.jsonl
#     # human-eval expects one JSON per sample: task_id + completion
#     with out_path.open("w", encoding="utf-8") as f:
#         for i in range(len(problems)):
#             task_id = problems[i].get("task_id", f"HumanEval/{i}")
#             prompt = problems[i]["prompt"]

#             # IMPORTANT: for HumanEval, feeding the raw prompt is usually best.
#             # (It includes the function signature + docstring; model completes the body.)
#             outputs = llm.generate([prompt], sampling_params)[0].outputs  # list length = n_samples

#             for o in outputs:
#                 completion = _postprocess_completion(o.text)
#                 row = {"task_id": task_id, "completion": completion}
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     print(f"[OK] Wrote samples: {out_path.resolve()}")

#     # 4) Run execution-based evaluation (official harness)
#     if run_eval:
#         # Requires: pip install -e human-eval (see openai/human-eval README)
#         cmd = ["python", "-m", "human_eval.evaluate_functional_correctness", str(out_path)]
#         print("[RUN]", " ".join(cmd))
#         subprocess.run(cmd, check=False)


# if __name__ == "__main__":
#     main()


#######################################
############version-gemma2#############

# coding=utf-8
# """
# Evaluate a model on HumanEval (execution-based) using openai/human-eval harness.

# Fixes applied (based on your GSM8K-stable setup):
# 1) Use chat template (for instruct models like Gemma2-it) to reduce chatty outputs.
# 2) Remove the forced leading "\n" in completion (can break indentation).
# 3) Prompt-aware postprocess: if prompt ends with indentation, strip leading newlines from completion.
# 4) Relax stop strings: keep only code fence stop ("```") to avoid truncating valid code.
# """

# import json
# import re
# import subprocess
# from pathlib import Path
# from typing import Optional

# import click
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams


# def _strip_code_fences(s: str) -> str:
#     """Remove ```python ... ``` fences if model emits them."""
#     s = s.strip()
#     s = re.sub(r"^\s*```(?:python)?\s*\n", "", s, flags=re.IGNORECASE)
#     s = re.sub(r"\n\s*```\s*$", "", s)
#     return s.rstrip()


# def _is_indent_suffix(prompt: str) -> bool:
#     """
#     True if prompt ends with whitespace indentation on the last line.
#     Common in HumanEval prompts where the last line is '    ' (4 spaces).
#     """
#     # Grab trailing whitespace at the very end
#     m = re.search(r"([ \t]+)$", prompt)
#     if not m:
#         return False
#     # If there's a newline somewhere, ensure the whitespace is on the last line
#     # (i.e., it is indentation, not arbitrary trailing spaces earlier)
#     # This is already implied by $.
#     return True


# def _postprocess_completion(prompt: str, completion: str) -> str:
#     """
#     HumanEval expects 'completion' to be appended to the provided prompt.

#     Key rule:
#       - DO NOT force a leading newline (can break indentation when prompt ends with '    ').
#       - If prompt ends with indentation, strip leading newlines from completion.
#       - Strip code fences.
#     """
#     c = _strip_code_fences(completion).rstrip()

#     # If prompt ends with indent spaces, leading newline can break indentation structure.
#     if _is_indent_suffix(prompt):
#         c = c.lstrip("\n")

#     # Also drop any leading empty lines (generally safe for HumanEval)
#     # but keep indentation if present.
#     # (We only strip newlines, not spaces.)
#     c = c.lstrip("\n")

#     return c


# def _build_humaneval_input(
#     prompt: str,
#     tokenizer: Optional[AutoTokenizer],
#     use_chat_template: bool,
#     system_prompt: str,
# ) -> str:
#     """
#     For instruct models, wrapping the raw HumanEval prompt in a chat template
#     tends to reduce chatty text and keep code-only outputs.
#     """
#     if not use_chat_template or tokenizer is None:
#         return prompt

#     # Keep it short and strict: "output only code continuation"
#     user_msg = (
#         "Complete the following Python function. "
#         "Return ONLY valid Python code that continues the given prompt. "
#         "Do not include explanations, markdown, or code fences.\n\n"
#         f"{prompt}"
#     )

#     msgs = []
#     if system_prompt:
#         msgs.append({"role": "system", "content": system_prompt})
#     msgs.append({"role": "user", "content": user_msg})

#     return tokenizer.apply_chat_template(
#         msgs,
#         tokenize=False,
#         add_generation_prompt=True,
#     )


# @click.command()
# @click.option("--ckpt", type=str, required=True, help="Model checkpoint/path for vLLM.")
# @click.option("--tokenizer", type=str, default=None, help="Tokenizer name/path (optional).")
# @click.option("--max_tokens", type=int, default=768)
# @click.option("--temperature", type=float, default=0.2)
# @click.option("--top_p", type=float, default=0.95)
# @click.option("--n_samples", type=int, default=1, help="How many samples per task (for pass@k).")
# @click.option("--out", type=str, default="humaneval_samples.jsonl")
# @click.option("--run_eval", type=bool, default=True, help="Run human-eval after generation.")
# @click.option(
#     "--use_chat_template",
#     type=bool,
#     default=True,
#     help="Wrap HumanEval prompt with chat template (recommended for instruct models).",
# )
# @click.option(
#     "--system_prompt",
#     type=str,
#     default="You are a helpful coding assistant.",
#     help="System prompt used when --use_chat_template=True.",
# )
# def main(
#     ckpt,
#     tokenizer,
#     max_tokens,
#     temperature,
#     top_p,
#     n_samples,
#     out,
#     run_eval,
#     use_chat_template,
#     system_prompt,
# ):
#     # 1) Load HumanEval problems
#     ds = load_dataset("openai/openai_humaneval")  # 164 tasks
#     split = "test" if "test" in ds else list(ds.keys())[0]
#     problems = ds[split]

#     # 2) Tokenizer (needed only for chat template)
#     tok = None
#     if use_chat_template:
#         tok = AutoTokenizer.from_pretrained(tokenizer or ckpt)

#     # 3) vLLM setup
#     sampling_params = SamplingParams(
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         n=n_samples,
#         # For code, keep stops minimal to avoid truncation.
#         stop=["```"],
#     )
#     llm = LLM(model=ckpt, tokenizer=tokenizer or ckpt)

#     out_path = Path(out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # 4) Generate + write samples.jsonl
#     with out_path.open("w", encoding="utf-8") as f:
#         for i in range(len(problems)):
#             task_id = problems[i].get("task_id", f"HumanEval/{i}")
#             raw_prompt = problems[i]["prompt"]

#             model_inp = _build_humaneval_input(
#                 prompt=raw_prompt,
#                 tokenizer=tok,
#                 use_chat_template=use_chat_template,
#                 system_prompt=system_prompt,
#             )

#             outputs = llm.generate([model_inp], sampling_params)[0].outputs

#             for o in outputs:
#                 completion = _postprocess_completion(raw_prompt, o.text)
#                 row = {"task_id": task_id, "completion": completion}
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     print(f"[OK] Wrote samples: {out_path.resolve()}")

#     # 5) Run execution-based evaluation (official harness)
#     if run_eval:
#         cmd = ["python", "-m", "human_eval.evaluate_functional_correctness", str(out_path)]
#         print("[RUN]", " ".join(cmd))
#         subprocess.run(cmd, check=False)


# if __name__ == "__main__":
#     main()