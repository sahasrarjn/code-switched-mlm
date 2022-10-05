# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
OUT_DIR=${4:-$REPO/PretrainedModels/en_hi_switch}
TRAIN_FILE=${5:-$REPO/Code/taggedData/en_hi_switch_train.txt}
EVAL_FILE=${6:-$REPO/Code/taggedData/en_hi_switch_eval.txt}
MLM_PROBABILITY=${7:-0.32}

# export NVIDIA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=${8:-2}
export WANDB_DISABLED="true"

EPOCH=4
BATCH_SIZE=4
MAX_SEQ=256

function getMLMprob {
    file=$1
    mask_cnt=$(grep -o -i MASK ${file} | wc -l)
    nomask_cnt=$(grep -o -i NOMASK ${file} | wc -l)
    total_tokens=$(python3 -c "print($mask_cnt + $nomask_cnt)")
    echo $(python3 -c "print( 0.15 * $total_tokens / $mask_cnt)" )
}

# TODO: use this function to automate


# Baseline MLM
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_baseline'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_baseline_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_baseline_eval.txt'
# MLM_PROBABILITY=0.15

# Standard Switch MLM
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_switch'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_eval.txt'
# MLM_PROBABILITY=0.32

# FreqMLM
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_freq'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_train.txt'
# MLM_PROBABILITY=0.33

# Experiment: FreqMLM-Complement
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_freq_complement'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_complemen_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_complemen_eval.txt'
# MLM_PROBABILITY=0.273

# Experiment: SwitchMLM-Complement
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_switch_complement'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_complemen_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_complemen_eval.txt'
# MLM_PROBABILITY=0.283

# Experiment: Inverted LID SwitchMLM
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_switch_inverted'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_inverted_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_switch_inverted_eval.txt'
# MLM_PROBABILITY=0.32

# Experiment: Maskable OTHER tokens (on top of FreqMLM)
# OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_freq_maskableOTHER'
# TRAIN_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_maskableOTHER_train.txt'
# EVAL_FILE='/home/sahasra/pretraining/Code/taggedData/en_hi_freq_maskableOTHER_eval.txt'
# MLM_PROBABILITY=0.234

# Experiment: Incorporate AMB tokens (on top of FreqMLM)
OUT_DIR='/home/sahasra/pretraining/PretrainedModels/en_hi_freq_AMB'
TRAIN_FILE='/home/sahasra/pretraining/Code/experiments/amb-tokens/en_hi_freq_amb_train.txt'
EVAL_FILE='/home/sahasra/pretraining/Code/experiments/amb-tokens/en_hi_freq_amb_eval.txt'
# --amb_tokens
# --mask0_probability 0.23
# --mask1_probability 0.2
# --mask2_probability 0.03


echo "Starting Pretraining With:"
echo "Train: $TRAIN_FILE"
echo "Eval: $EVAL_FILE"
echo "Output Here: $OUT_DIR"

python3.6 $PWD/Code/utils/pretrain.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE \
    --config_name $MODEL   \
    --tokenizer_name  $MODEL \
    --output_dir $OUT_DIR \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $EVAL_FILE \
    --mlm \
    --line_by_line \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 20\
    --num_train_epochs $EPOCH\
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --mlm_probability $MLM_PROBABILITY \
    --amb_tokens \
    --mask0_probability 0.23 \
    --mask1_probability 0.2 \
    --mask2_probability 0.03

echo "Find Output Here: $OUT_DIR"

