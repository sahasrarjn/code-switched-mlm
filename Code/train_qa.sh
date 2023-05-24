# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-QA_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}
export CUDA_VISIBLE_DEVICES=0

BATCH_SIZE=4 #1
EVAL_BATCH_SIZE=4 #2
MAX_SEQ=256


language='Hindi'
pretrainModel='en_hi_baseline'


for seed in 32
	do
	echo "SEED: $seed"
	for ep in 20
    do
		echo "EPOCHS: $ep"
		for mod in "PretrainedModels/${language}/${pretrainModel}/checkpoint-*"
		do
		echo $mod
		python3.6 $PWD/Code/utils/run_qa.py \
			--data_dir $DATA_DIR/$TASK \
			--output_dir $OUT_DIR/$TASK \
			--model_type $MODEL_TYPE \
			--model_name_or_path $MODEL \
			--do_eval \
			--save_stats_file logs/train_qa/all \
			--num_train_epochs $ep \
			--evaluate_during_training \
			--per_gpu_train_batch_size $BATCH_SIZE \
			--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
			--max_seq_length $MAX_SEQ \
			--overwrite_output_dir \
			--seed $seed \
			--logging_steps 7000 \
			--gradient_accumulation_steps 10 \
			--do_train \
			--train_file train-v2.0.json \
			--predict_file dev-v2.0.json \
			--model_loc $mod 
    	done
  	done 
done
