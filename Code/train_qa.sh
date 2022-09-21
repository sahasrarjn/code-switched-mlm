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
export NVIDIA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=3

EPOCH=4
BATCH_SIZE=4 #1
MLM_BATCH_SIZE=4 #1
EVAL_BATCH_SIZE=4 #2
MAX_SEQ=256
MLM_MAX_SEQ=256
PRETR_EPOCH=3

# for seed in 32 42 52 62 72 82 92 102 112 122
for seed in 22
	do
	# for ep in 6 7 8 9
	# for ep in 20 30 40
	for ep in 1
    do
		# for mod in "Trial_TCS_MLM/checkpoint-24720" 
		for mod in "PretrainedModels/freq_en_hi2/checkpoint-7680"
		do
		echo $mod
		python3.6 $PWD/Code/utils/run_squad_vanilla.py \
			--data_dir $DATA_DIR/$TASK \
			--output_dir $OUT_DIR/$TASK \
			--model_type $MODEL_TYPE \
			--model_name_or_path $MODEL \
			--do_eval \
			--num_train_epochs $ep \
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
			--save_stats_file all_8seeds_bert_switch_mlm_realTCS \
			--model_loc $mod
    	done
  	done 
done
