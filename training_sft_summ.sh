export PYTHONPATH=$PYTHONPATH:/path/to/alignment-handbook/src/
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero3.yaml train/train_sft.py config/sft/sft_config_summ.yaml