# Veto: Stable On-Policy Distillation through Adaptive Target Reformulation

📄 Paper: [arXiv:2601.07155](https://arxiv.org/abs/2601.07155)

Official implementation of **Veto**, an objective-level reformulation for on-policy
knowledge distillation (KD) that stabilizes optimization by constructing a geometric
bridge between the teacher and the student in logit space.

Instead of forcing an early-stage student to match the teacher directly, Veto builds an
intermediate target distribution

```
Q(y|x) ∝ exp( z_T(y|x) + β · z_S(y|x) )  =  (1/Z) · P_T(y|x) · P_S(y|x)^β
```

a *Product of Experts* that keeps probability mass only where **both** the teacher
(quality) and the student (confidence) agree. A single scalar `β` plays a dual role:

- **Forward KL** — `β` acts as an *Adaptive Gradient Veto*, suppressing pathological
  gradients on tokens where the student is ignorant (`P_S → 0`).
- **Reverse KL** — `β` acts as a *Decisiveness Knob*, trading off mode-seeking
  decisiveness against output diversity (bridging KD and REINFORCE).

The core reformulation lives in `blend_teacher_logps()` / `get_scheduled_B()` in
[train/ddp_veto.py](train/ddp_veto.py).

---

## Pipeline Overview

The full reproduction is a two-stage process:

1. **Teacher SFT** — supervise the large teacher (e.g. Qwen2-7B-Instruct) on the
   task data using [alignment-handbook](https://github.com/huggingface/alignment-handbook).
2. **Student KD** — distill the small student (e.g. Qwen2-0.5B-Instruct) from the
   SFT'd teacher using Veto (on-policy by default).
3. **Evaluation** — generate with the trained student (vLLM) and score per task.

```
Teacher (Qwen2-7B-IT)  ──SFT──▶  Teacher-SFT
                                     │
Student (Qwen2-0.5B-IT) ──on-policy KD (Veto)──▶  Student-Veto  ──eval──▶ metrics
```

---

## Environment Setup

We use **two** environments because of conflicting dependencies.

### 1. Teacher SFT environment (alignment-handbook)

```bash
git clone -b v0.3-release https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .
python -m pip install flash-attn --no-build-isolation
```

### 2. `veto` — Student KD training **and** inference

This single environment is enough to train **and** run inference for our method (Veto).

```bash
conda create -n veto python=3.10 -y
conda activate veto
pip3 install torch
pip3 install -r requirements.txt
pip install -U "transformers==4.57.3"
```

> **Optional (SKD baseline only):** to reproduce the Speculative Knowledge Distillation
> (SKD) baseline, its interleaved sampling needs a patched `transformers` generation
> module:
> ```bash
> cp transformers/* /path/to/envs/veto/lib/python3.10/site-packages/transformers/generation/
> ```

> For GPT-4o-mini-judged summarization evaluation (`eval/eval_summ.py`), also set
> `export OPENAI_API_KEY=...`.

---

## Data

The task data ships under [data/](data/). Each task expects JSON files of the
form `data/<task_type>_train.json` / `data/<task_type>_vali.json`.

| Stage           | Data location                                  |
| --------------- | ---------------------------------------------- |
| Teacher SFT     | `data/gsm_8k/`, `data/wizardcoder_evol_10k_*`, `data/summ_1k_*` |
| Student KD      | `data/<task_type>_train.json` (1K sampled instances) |
| Validation      | `data/<task_type>_vali.json`                    |

---

## Step 1 — Teacher Supervised Fine-Tuning (SFT)

SFT is run through alignment-handbook with the configs in [config/sft/](config/sft/).
Edit `model_name_or_path`, `dataset_mixer`, and `output_dir` for your setup
(see [config/sft/sft_config_example.yaml](config/sft/sft_config_example.yaml) for the
documented template).

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/alignment-handbook/src/
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file config/deepspeed_zero3.yaml \
  train/train_sft.py \
  config/sft/sft_config_gsm.yaml
```

Per-task SFT launchers (edit the alignment-handbook path inside each):

| Task            | Config                                  | Launcher                       |
| --------------- | --------------------------------------- | ------------------------------ |
| GSM8K (teacher) | `config/sft/sft_config_gsm.yaml`        | `training_sft_gsm.sh`          |
| Code            | `config/sft/sft_config_code.yaml`       | `training_sft_code.sh`         |
| Summarization   | `config/sft/sft_config_summ.yaml`       | `training_sft_summ.sh`         |

---

## Step 2 — Student Knowledge Distillation (Veto)

KD is configured via a YAML file in [config/](config/) and launched through
[train/run_kd_train.py](train/run_kd_train.py), which assembles the `accelerate` command
and calls [train/ddp_veto.py](train/ddp_veto.py).

```bash
# uses config/kd_train.yaml by default
python train/run_kd_train.py config/kd_train_code.yaml
```

Before launching, edit the config:

```yaml
model_params:
  checkpoint_template: ./checkpoints/Qwen2-7B-Instruct-gsm_7k-sft   # teacher SFT ckpt
  assistant_checkpoint_template: Qwen/Qwen2-0.5B-Instruct           # student init
  tokenizer_name: Qwen/Qwen2-7B-Instruct

resource_params:
  gpu_group: "0,1"           # CUDA_VISIBLE_DEVICES
  num_processes: 2
  user: your_username
  wandb_key: YOUR_WANDB_API_KEY
  wandb_proj: your_project
```

Per-task KD configs / launchers:

| Task          | `task_type` | KD config                       | Launcher              |
| ------------- | ----------- | ------------------------------- | --------------------- |
| GSM8K         | `gsm_1k`    | `config/kd_train.yaml`          | `training_gsm.sh`     |
| Code          | `code`      | `config/kd_train_code.yaml`     | `training_code.sh`    |
| Summarization | `summ_1k`   | `config/kd_train_summ.yaml`     | `training_summ.sh`    |

> **Note:** Avoid running `accelerate` under `nohup`; it can cause unexpected crashes.

### Key hyperparameters

| Group              | Param                                    | Meaning |
| ------------------ | ---------------------------------------- | ------- |
| `kd_params`        | `kd_type`                                | `on-policy` (main), `supervised_kd`, `skd`, `mixed`, `seq_kd` |
|                    | `distance_metric`                        | `kl` = forward KL (Adaptive Gradient Veto), `reverse_kl` = reverse KL (Decisiveness Knob), `jsd` |
|                    | **`B_start` / `B_end`**                  | the Veto parameter **β** at the start / end of training |
|                    | **`B_schedule`**                         | `linear` (decay β_start→β_end, recommended) or `const` |
|                    | `top_k`                                  | top-k for SKD only (e.g. 25); keep 0 otherwise |
|                    | `student_temperature`, `student_top_p`   | on-policy sampling from the student |
|                    | `teacher_temperature`, `teacher_top_p`   | teacher decoding |
| `task_params`      | `inp_length`, `max_new_tokens`           | prompt / generation lengths (`max_length` is their sum) |
| `training_params`  | `lr`, `num_epoch`, `grad_acc_size`, `seed`, `eval_step`, `early_stop_epoch`, `mixed_precision` | `lr=1e-5`, `num_epoch=3`, `bf16` |
| `exec_params`      | `enable_stop_token`, `ckpt_prefix`, `debug_enable` | stop-token trimming, run name, debug mode |

### Recommended β (from the paper)

A high β stabilizes the early "ignorant" phase; linearly decaying it to 0 lets the
student capture the teacher's finer structure later. Best results use forward KL with
**linear β decay** starting from:

| Task            | β (start) | Schedule | Divergence |
| --------------- | --------- | -------- | ---------- |
| Reasoning (GSM) | 0.8       | linear   | forward KL |
| Code            | 1.0       | linear   | forward KL |
| Summarization   | 0.3       | linear   | forward KL |

> The committed YAML files reflect specific experimental runs; set `B_start`,
> `B_end`, `B_schedule`, and `distance_metric` to match the table above to reproduce
> the main results.

---

## Step 3 — Evaluation

Evaluation uses [vLLM](https://github.com/vllm-project/vllm) (`pip install vllm`); point
`-ckpt` at a saved student checkpoint.

```bash
# GSM8K (answer accuracy, openai/gsm8k test)
python eval/eval_gsm.py -max_tokens 512 -ckpt /path/to/student-ckpt

# HumanEval (pass@k, execution-based via openai/human-eval)
python eval/eval_code.py --ckpt /path/to/student-ckpt --tokenizer Qwen/Qwen2-7B-Instruct \
  --max_tokens 768 --n_samples 10

# Summarization win-rate (GPT-4o-mini judge; needs OPENAI_API_KEY)
python eval/eval_summ.py -max_tokens 128 -ckpt /path/to/student-ckpt
```

The numeric answer grader used by GSM8K is [eval/grader.py](eval/grader.py).

---

## Repository Structure

```
veto/
├── config/
│   ├── deepspeed_zero3.yaml        # accelerate / DeepSpeed ZeRO-3 config
│   ├── kd_train*.yaml              # KD configs (one per task)
│   └── sft/                        # SFT configs (alignment-handbook)
├── data/                           # task JSON splits
├── train/
│   ├── train_sft.py                # teacher SFT entry point
│   ├── run_kd_train.py             # KD launcher (builds accelerate command)
│   └── ddp_veto.py                 # core KD + Veto training loop  ★
├── eval/
│   ├── eval_gsm.py                 # GSM8K accuracy (vLLM)
│   ├── eval_code.py                # HumanEval pass@k
│   ├── eval_summ.py             # DialogSum win-rate (GPT-4o-mini judge)
│   ├── grader.py                   # numeric answer grader (GSM8K)
│   └── vali_loss_compute.py        # validation-loss helper
├── transformers/                   # patched generation files (SKD only)
├── training_*.sh                   # per-task launchers
└── requirements.txt                # Python dependencies
```

`_to_review/` collects clearly redundant / scratch files (old eval variants, backups)
that were moved out of the main tree for public release — review and delete as desired.

---

## Citation

```bibtex
@inproceedings{jang2026veto,
  title     = {Stable On-Policy Distillation through Adaptive Target Reformulation},
  author    = {Jang, Ijun and Yeom, Jewon and Yeo, Juan and Lim, Hyunggu and Kim, Taesup},
  year      = {2026},
  eprint    = {2601.07155},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2601.07155}
}
```

### Acknowledgements

The on-policy / speculative KD scaffolding builds on
[Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) (Xu et al., 2025).
