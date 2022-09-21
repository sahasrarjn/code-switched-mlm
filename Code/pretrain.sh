# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
ROMA="romanised"
OUT_DIR=${5:-"$REPO/Results"}
MLM_DATA_FILE=${6:-"$REPO/ishan_data/trial.txt"}
export NVIDIA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=3

EPOCH=4
BATCH_SIZE=4
MAX_SEQ=256

echo "Starting Pretraining"

python3.6 $PWD/Code/utils/pretrain.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE \
    --config_name $MODEL   \
    --tokenizer_name  $MODEL \
    --output_dir $REPO/PretrainedModels/freq_en_hi_try1 \
    --train_data_file $REPO/Code/taggedData/en_hi_freq_train.txt \
    --eval_data_file $REPO/Code/taggedData/en_hi_freq_eval.txt \
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
    --mlm_probability 0.32