# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

MODEL_NAME=${1:-bert}
MODEL=${2:-/home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160}
MODEL_CONDITIONAL=${3:-None}
MODEL_TYPE=${4:-bilstm-head}
OUT_DIR=${5:-/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline}
EXP_NAME=${6:-baseline-bilstm-probing}
TRAIN_FILE=${7:-/home/sahasra/pretraining/Code/experiments/probing-task/probe-target-switch-points_train.txt}
EVAL_FILE=${8:-/home/sahasra/pretraining/Code/experiments/probing-task/probe-target-switch-points_eval.txt}

export CUDA_VISIBLE_DEVICES=${9:-1}

EPOCH=1
PROBE_LAYER=-1

# MODEL_TYPES:
# bilstm-head or linear-head or conditional-linear-probing

echo "Starting Probing With:"
echo "Train: $TRAIN_FILE"
echo "Eval: $EVAL_FILE"
echo "Output Here: $OUT_DIR"

python3.6 $PWD/Code/utils/probing.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL \
    --model_conditional_name $MODEL_CONDITIONAL_NAME \
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
    --probe_layer $PROBE_LAYER

echo "Find Output Here: $OUT_DIR"

