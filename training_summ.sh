#!/usr/bin/env bash
# # environment setup
# # source ${HOME}/.bashrc
# # source ${HOME}/anaconda3/bin/activate
# # eval "$(conda shell.bash hook)"
# # conda activate veto               # change to your conda env
# #export PYTHONPATH=$PYTHONPATH:/home/ijunjang/neco/NeCo
# python train/run_kd_train.py config/kd_train.yaml

EXP_NAME="supervised_kd-contrastive-add-0_0-summ_1k" #"on-policy-contrastive-add-0_05-summ_1k" #-contrastive-add-cka-epoch6-7k" #_cka_1e-1_gsm_8k"
LOG_FILE="logs/${EXP_NAME}.log"

echo "[INFO] experiment: ${EXP_NAME}"
echo "[INFO] log file: ${LOG_FILE}"

python train/run_kd_train.py config/kd_train_summ.yaml >"$LOG_FILE" 2>&1