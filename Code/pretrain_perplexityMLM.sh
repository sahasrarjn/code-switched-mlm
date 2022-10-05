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

function getMLMprob {
    file=$1
    mask_cnt=$(grep -o -i MASK ${file} | wc -l)
    nomask_cnt=$(grep -o -i NOMASK ${file} | wc -l)
    total_tokens=$(python3 -c "print($mask_cnt + $nomask_cnt)")
    echo $(python3 -c "print( 0.15 * $total_tokens / $mask_cnt)" )
}


echo "starting perplexity update"

echo "Computing model scores"
python3.6 $PWD/Code/utils/PerplexityMasking/compute_model_score.py \
    -m bert-base-multilingual-cased \
    -l $PWD/PretrainedModels/en_hi_baseline/final_model \
    -f $PWD/Data/MLM/combined/combined.txt \
    -o $PWD/Code/utils/PerplexityMasking/data/train_scores.txt \
    -d cuda
    # -f $PWD/Code/utils/PerplexityMasking/real_CS_data/spanish_switch_sents.txt \

echo "Creating masks"
python3.6 $PWD/Code/utils/PerplexityMasking/create_masks.py \
    -s $PWD/Code/utils/PerplexityMasking/data/train_scores.txt \
    -i $PWD/Data/MLM/combined/combined.txt \
    -o $PWD/Code/utils/PerplexityMasking/data/all.txt \

echo "Generating train / eval file"
python3.6 $PWD/Code/utils/PerplexityMasking/split_train_eval.py \
    -f $PWD/Code/utils/PerplexityMasking/data/all.txt \
    -t $PWD/Code/utils/PerplexityMasking/data/train.txt \
    -e $PWD/Code/utils/PerplexityMasking/data/eval.txt \
    --frac 0.1

file="$PWD/Code/utils/PerplexityMasking/data/all.txt"
mlm_probability=$(getMLMprob ${file})
echo "MLM probability: ${mlm_probability}"

echo "starting pretraining"
python3.6 $PWD/Code/utils/Bert_custom_MLM.py \
    --model_name_or_path $REPO/PretrainedModels/en_hi_baseline \
    --model_type $MODEL_TYPE \
    --tokenizer_name  $MODEL \
    --output_dir $REPO/PretrainedModels/en_hi_perp \
    --train_data_file $REPO/Code/utils/PerplexityMasking/data/train.txt \
    --eval_data_file $REPO/Code/utils/PerplexityMasking/data/eval.txt \
    --mlm \
    --line_by_line \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 20\
    --num_train_epochs $EPOCH\
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --mlm_probability $mlm_probability