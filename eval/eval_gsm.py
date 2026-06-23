# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a model on the GSM8K dataset."""
import math
import json
import click
from datasets import load_dataset
from grader import math_equal
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams
import re
def _sanitize_number_expr(s: str) -> str:
  """Extract the final numerical expression from model output reliably."""

  # Remove \boxed{}, $, whitespace
  s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
  s = s.replace('$', '').strip()

  # Remove thousand separators: 1,234 -> 1234
  s = re.sub(r'(?<=\d),(?=\d)', '', s)

  # 1) handle the "The answer is: 230" form (most important!)
  m_ans = re.search(
      r'(?:the|final)?\s*answer\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?)',
      s, flags=re.IGNORECASE)
  if m_ans:
    return m_ans.group(1).strip()

  # 2) trailing number as in "Therefore, the answer is 15."
  m_lastnum = re.search(r'(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?)\s*\.?\s*$', s)
  if m_lastnum:
    return m_lastnum.group(1).strip()

  # 3) Tuple -> use only the last value
  m_tuple = re.search(r'\(([^()]*)\)', s)
  if m_tuple:
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', m_tuple.group(1))
    nums = [n.strip() for n in nums if n.strip() != '']
    if nums:
      return nums[-1]

  # 4) extract all numbers and use only the last one
  nums = re.findall(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', s)
  nums = [n.strip() for n in nums if n.strip() != '']
  if nums:
    return nums[-1]

  # No number found
  return s.strip()


@click.command()
@click.option('-max_tokens', type=int, default=512)
@click.option('-visualize_text', type=bool, default=True)
@click.option('-ckpt', type=str, default='gemma-7b-it-gsm-1k')
def main(max_tokens, visualize_text, ckpt):
  # load in validation set
    vali_dataset = load_dataset('openai/gsm8k', 'main')['test']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    #tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    vali_data_inp = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f'Q: {ele} \n\nA:'}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ele in vali_dataset['question']
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0, top_p=1, stop=['\n\n']
    )
    llm = LLM(model=ckpt,    tokenizer="Qwen/Qwen2-7B-Instruct")
    gen_outputs = llm.generate(vali_data_inp, sampling_params)
    count = 0
    total = 0
    wrong_cases=[]
    for idx, (_, ele, ref) in enumerate(zip(vali_data_inp, gen_outputs, vali_dataset['answer'])):
        text = ele.outputs[0].text
        if '####' in ele.outputs[0].text:
            ans_raw = ele.outputs[0].text.split('####',1)[1].split('\n',1)[0].strip()
            gt_raw = ref.split('####',1)[1].strip()
        else:
            ans_raw=ele.outputs[0].text
            gt_raw = ref.split('####',1)[1].strip()

        ans=_sanitize_number_expr(ans_raw)
        gt  = _sanitize_number_expr(gt_raw)
        print(ans)
        print(gt)
        try:
            if r'\pi' in ans or r'\pi' in gt:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(ans, gt, timeout=True, pi=pi))
                equiv = any(equivs)
            else:
                equiv = math_equal(ans, gt, timeout=True)
        except (ValueError, TypeError) as error:
            equiv = False
            print(error)
        if equiv:
            count += 1
        else:
            if visualize_text:
                print(ele.outputs[0].text)
                print('-' * 50)
                print(ref)
                print('>' * 50)
        total += 1

        if not equiv:
            wrong_cases.append({
                "idx": idx,
                "question": vali_dataset['question'][idx],
                "model_answer": ans,
                "gold_answer": gt,
                "raw_output": ele.outputs[0].text
            })
    print('Acc: ', count / total)
    #if failures_path:
    with open("failure.json", 'w', encoding='utf-8') as f:
        for row in wrong_cases:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
  main()