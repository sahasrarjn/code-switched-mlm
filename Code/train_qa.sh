# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-QA_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}
# MLM_DATA_FILE=${6:-"$REPO/Data/MLM/all_combined/train_roman.txt"}
MLM_DATA_FILE=${6:-"$REPO/ishan_data/ishan_plus_65k.txt"}
# MLM_DATA_FILE=${6:-"$REPO/ishan_data/ishan_plus_65k_plus_qa.txt"}
# export NVIDIA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=0

# EPOCH=4
BATCH_SIZE=4 #1
# MLM_BATCH_SIZE=4 #1
EVAL_BATCH_SIZE=4 #2
MAX_SEQ=256
# MLM_MAX_SEQ=256
# PRETR_EPOCH=3

# wandb login 98d0804992a30ee86b8971c931bffcfeff2d5640

language='Hindi'
pretrainModel='en_hi_switch_inverted_aux_new'

# To run Residual Bert experiments:
# Code/./train_qa.sh QA_EN_HI bert-base-multilingual-cased residual-bert_4_0.2

for seed in 32 42 52 62 72 82 92 102 112 122
	do
	echo "SEED: $seed"
	for ep in 20 30 40
    do
		echo "EPOCHS: $ep"
		for mod in "PretrainedModels/${language}/${pretrainModel}/checkpoint-*"
		do
		echo $mod
		python3.6 $PWD/Code/utils/run_squad_vanilla.py \
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
			--model_loc $mod \
			# --wandb \
			# --experiment-name "${pretrainModel}_seed${seed}_epoch${ep}"
    	done
  	done 
done
