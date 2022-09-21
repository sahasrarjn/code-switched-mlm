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

echo "starting perplexity update"

python3.6 $PWD/Code/utils/Bert_custom_MLM.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE \
    --tokenizer_name  $MODEL \
    --config_name $MODEL   \
    --output_dir $REPO/PretrainedModels/bert_realcs_perpupd_seed100_hindi \
    --train_data_file $REPO/Code/utils/PerplexityMasking/tmp/train.txt \
    --eval_data_file $REPO/Code/utils/PerplexityMasking/tmp/eval.txt \
    --mlm \
    --first \
    --line_by_line \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 20\
    --num_train_epochs 1\
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --mlm_probability 0.3

exit
for e in 2 3 4
do
echo $e
python3.6 $PWD/Code/PerplexityMasking/compute_model_score.py \
-m bert-base-multilingual-cased \
-l $PWD/PretrainedModels/bert_realcs_perpupd_seed100_hindi/final_model \
-f $PWD/Data/MLM/combined/combined.txt \
-o $PWD/Code/PerplexityMasking/tmp/train_scores.txt \
-d cuda \
# -f $PWD/Code/PerplexityMasking/real_CS_data/spanish_switch_sents.txt \

python3.6 $PWD/Code/PerplexityMasking/create_masks.py

python3.6 $PWD/Code/PerplexityMasking/split_train_eval.py

echo "starting pretraining"

python3.6 $PWD/Code/utils/Bert_custom_MLM.py \
    --model_name_or_path $REPO/PretrainedModels/bert_realcs_perpupd_seed100_hindi \
    --model_type $MODEL_TYPE \
    --tokenizer_name  $MODEL \
    --output_dir $REPO/PretrainedModels/bert_realcs_perpupd_seed100_hindi \
    --train_data_file $REPO/Code/PerplexityMasking/tmp/train.txt \
    --eval_data_file $REPO/Code/PerplexityMasking/tmp/eval.txt \
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
    --mlm_probability 0.3 
    # --num_train_epochs 1\
done