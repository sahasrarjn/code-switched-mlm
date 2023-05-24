# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
OUT_DIR=${4:-$REPO/PretrainedModels/en_hi_baseline}
TRAIN_FILE=${5:-$REPO/Code/taggedData/en_hi_baseline_train.txt}
EVAL_FILE=${6:-$REPO/Code/taggedData/en_hi_baseline_eval.txt}
MLM_PROBABILITY=$(getMLMprob Code/taggedData/en_hi_baseline.txt)

export CUDA_VISIBLE_DEVICES=${8:-3}
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

echo "Starting Pretraining With:"
echo "Train: $TRAIN_FILE"
echo "Eval: $EVAL_FILE"
echo "Output Here: $OUT_DIR"
echo "MLM prob: $MLM_PROBABILITY"

python3.6 $PWD/Code/utils/pretrain_auxloss.py \
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
    --weight_decay 5e-4 \
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --mlm_probability $MLM_PROBABILITY \
    --overwrite_output_dir \
    --res_layer 2 \
    --res_dropout 0.5

echo "Find Output Here: $OUT_DIR"

