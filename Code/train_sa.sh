# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-Sentiment_EN_HI/Romanized}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

BATCH_SIZE=4
MAX_SEQ=256


dir=`basename "$TASK"`

if [ $dir == "Devanagari" ] || [ $dir == "Romanized" ]; then
  OUT=`dirname "$TASK"`
else
  OUT=$TASK
fi

# OUTPUT_DIR_FINE="new_switch_fine_spanishtry2"
OUTPUT_DIR_FINE="Results/switch_en_hi"


#for sst use 350 steps, for semeval use 100 steps, for ml finetue use 40
for seed in  32
# for seed in  32 42 52 62 72 82 92 102 112 122
do
	for epochs in 20
	do
		mod="PretrainedModels/switch_en_hi/final_model"
		saveFileStart="switch_en_hi"

		python3.6 $PWD/Code/utils/BertSequenceTrained.py \
		--data_dir $DATA_DIR/$TASK/gluecos \
		--output_dir $OUT_DIR/$OUT/$OUTPUT_DIR_FINE \
		--model_type $MODEL_TYPE \
		--model_name $MODEL \
		--num_train_epochs $epochs \
		--train_batch_size $BATCH_SIZE \
		--logging_steps 100 \
		--eval_batch_size $BATCH_SIZE \
		--save_steps 0 \
		--seed $seed  \
		--learning_rate 1.5e-5 \
		--do_train \
		--max_seq_length $MAX_SEQ \
		--gradient_accumulation_steps 5 \
		--model_loc $mod \
		--save_file_start $saveFileStart
	done
done