# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

MODEL=${1:-/home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160}
MODEL_CONDITIONAL=${2:-None}
MODEL_TYPE=${3:-bilstm-head}
OUT_DIR=${4:-/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline}
EXP_NAME=${5:-baseline-bilstm-probing}
TRAIN_FILE=${6:-/home/sahasra/pretraining/Code/experiments/probing-task/probe-target-switch-points_train.txt}
EVAL_FILE=${7:-/home/sahasra/pretraining/Code/experiments/probing-task/probe-target-switch-points_eval.txt}

export CUDA_VISIBLE_DEVICES=${8:-2}

EPOCH=1

wandb login 98d0804992a30ee86b8971c931bffcfeff2d5640

# Baseline
# MODEL = /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160
# MODEL_CONDITIONAL = None
# OUT_DIR = /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline
# EXP_NAME = baseline-bilstm-probing or baseline-linear-probing
# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160 None linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-linear/ baseline-linear-probing &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-linear/log.file
####################################################

# SwitchMLM
# MODEL = /home/sahasra/pretraining/PretrainedModels/en_hi_switch/checkpoint-7920
# MODEL_CONDITIONAL = None
# OUT_DIR = /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch
# EXP_NAME = switch-bilstm-probing or switch-linear-probing
# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/en_hi_switch_inverted/checkpoint-7920 None linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-linear/ switch-linear-probing &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-linear/log.file
####################################################

# Conditional Probing
# MODEL = /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160
# MODEL_CONDITIONAL = /home/sahasra/pretraining/PretrainedModels/en_hi_switch/checkpoint-7920
# OR
# MODEL_CONDITIONAL = None
# OUT_DIR = /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-concat
# OR
# OUT_DIR = /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-zeros
# EXP_NAME = conditional-linear-probing-concat or conditional-linear-probing-zeros

# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160 /home/sahasra/pretraining/PretrainedModels/en_hi_switch_inverted/checkpoint-7920 conditional-linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-logging-10-concat/ conditional-linear-probing-concat &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-logging-10-concat/log.file
# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160 None conditional-linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-logging-10-zeros/ conditional-linear-probing-zeros &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-linear-logging-10-zeros/log.file
####################################################

# MODEL_TYPES:
# bilstm-head or linear-head or conditional-linear-probing

echo "Starting Probing With:"
echo "Train: $TRAIN_FILE"
echo "Eval: $EVAL_FILE"
echo "Output Here: $OUT_DIR"

python3.6 $PWD/Code/utils/probing.py \
    --model_path $MODEL \
    --model_conditional_path $MODEL_CONDITIONAL \
    --model_type $MODEL_TYPE \
    --output_dir $OUT_DIR \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $EVAL_FILE \
    --line_by_line \
    --do_train \
    --do_eval \
    --num_train_epochs $EPOCH\
    --seed 100 \
    --overwrite_output_dir \
    --logging_steps 10 \
    --max_steps 500 \
    --wandb \
    --experiment_name $EXP_NAME 

echo "Find Output Here: $OUT_DIR"

