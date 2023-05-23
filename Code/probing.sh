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

wandb login 98d0804992a30ee86b8971c931bffcfeff2d5640

# MODEL_NAME = bert

#### PROBING on intermediate layers ####
# Probing on 9th layer
# Baseline - Linear - Probing Layer 9
# MODEL_NAME=bert
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=baseline-linear-probing-layer9
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline/checkpoint-8160
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-linear-probing-layer9
# PROBE_LAYER=9

# Probing on 2nd layer
# Switch - Linear - Probing Layer 2
# MODEL_NAME=bert
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=switch-linear-probing-layer2
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-linear-probing-layer2
# PROBE_LAYER=2

# Probing on 9th layer
# Residual Baseline - Linear - Probing Layer 9
# MODEL_NAME=residual-bert_9_0.5
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_9_0.5-baseline-linear-probing-layer9
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline_residbert_9_0.5/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_9_0.5-baseline-linear-probing-layer9
# PROBE_LAYER=9

# Probing on 2nd layer
# Residual Switch - Linear - Probing Layer 2
# MODEL_NAME=residual-bert_2_0.5
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_2_0.5-switch-linear-probing-layer2
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_residbert_2_0.5/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_2_0.5-switch-linear-probing-layer2
# PROBE_LAYER=2

## Auxilary Loss version
# Probing on 2nd layer
# Residual Switch - Linear - Probing Layer 2 - Auxilary Loss
# MODEL_NAME=residual-bert_2_0.5
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_2_0.5-switch-linear-probing-aux-layer2
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted_aux/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_2_0.5-switch-linear-probing-aux-layer2
# PROBE_LAYER=2

# Probing on last layer
# Residual Switch - Linear - Probing Layer End - Auxilary Loss
# MODEL_NAME=residual-bert_2_0.5
# MODEL_CONDITIONAL_NAME=None
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_2_0.5-switch-linear-probing-aux
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted_aux/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_2_0.5-switch-linear-probing-aux
# PROBE_LAYER=-1
########################################

# Residual Baseline - Linear
# MODEL_NAME=residual-bert_4_0.5
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_4_0.5-baseline-linear-probing
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline_residbert_4_0.5/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_4_0.5-baseline-linear-probing

# Residual Switch - Linear
# MODEL_NAME=residual-bert_2_0.5
# MODEL_CONDITIONAL=None
# MODEL_TYPE=linear-head
# EXP_NAME=residual-bert_2_0.5-switch-linear-probing
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_residbert_2_0.5/checkpoint-7920
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_2_0.5-switch-linear-probing

# Residual Baseline - Conditional Linear
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline/checkpoint-8160
# MODEL_CONDITIONAL_NAME=residual-bert_4_0.5
# MODEL_CONDITIONAL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline_residbert_4_0.5/checkpoint-7920
# MODEL_TYPE=conditional-linear-head
# EXP_NAME=residual-bert_4_0.5-baseline-conditional-linear-probing-concat
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_4_0.5-baseline-conditional-linear-probing-concat

# Residual Switch - Conditional Linear
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline/checkpoint-8160
# MODEL_CONDITIONAL_NAME=residual-bert_2_0.5
# MODEL_CONDITIONAL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_residbert_2_0.5/checkpoint-7920
# MODEL_TYPE=conditional-linear-head
# EXP_NAME=residual-bert_2_0.5-switch-conditional-linear-probing-concat
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/residual-bert_2_0.5-switch-conditional-linear-probing-concat

# Baseline
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_baseline/checkpoint-8160
# MODEL_CONDITIONAL=None
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-mlp
# # EXP_NAME = baseline-bilstm-probing or baseline-linear-probing
# MODEL_NAME=bert
# MODEL_CONDITIONAL_NAME=None
# MODEL_TYPE=mlp-head
# EXP_NAME=baseline-mlp-probing
# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/baseline_en_hi/checkpoint-8160 None linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-linear/ baseline-linear-probing &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/baseline-linear/log.file
####################################################

# SwitchMLM
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted/checkpoint-7920
# MODEL_CONDITIONAL=None
# MODEL_NAME=bert
# MODEL_CONDITIONAL_NAME=None
# MODEL_TYPE=mlp-head
# EXP_NAME=switch-mlp-probing
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-mlp
# EXP_NAME = switch-bilstm-probing or switch-linear-probing
# Code/./probing.sh /home/sahasra/pretraining/PretrainedModels/en_hi_switch_inverted/checkpoint-7920 None linear-head /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-linear/ switch-linear-probing &> /home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/switch-linear/log.file
####################################################

# Conditional Probing 
# MODEL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted/checkpoint-7920
# MODEL_CONDITIONAL=/home/sahasra/pretraining/PretrainedModels/Hindi/en_hi_switch_inverted/checkpoint-7920
# MODEL_CONDITIONAL=None
# MODEL_NAME=bert
# MODEL_CONDITIONAL_NAME=bert
# MODEL_CONDITIONAL_NAME=None
# MODEL_TYPE=conditional-mlp-head
# EXP_NAME=conditional-mlp-zeros
# OUT_DIR=/home/sahasra/pretraining/Code/experiments/probing-task/temp-runs/conditional-mlp-zeros
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
    --wandb \
    --experiment_name $EXP_NAME  \
    --probe_layer $PROBE_LAYER

echo "Find Output Here: $OUT_DIR"

