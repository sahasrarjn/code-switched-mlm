# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
LANG=$1
TASK=${2:-Sentiment_EN_HI/Romanized}
MODEL=${3:-bert-base-multilingual-cased}
MODEL_TYPE=${4:-bert}
DATA_DIR=${5:-"$REPO/Data/Processed_Data"}
OUT_DIR=${6:-"$REPO/Results"}
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1

BATCH_SIZE=8
MAX_SEQ=256


if [ $LANG == "HI" ] 
then
	TASK="Sentiment_EN_HI/Romanized"
	pretrainedModel='en_hi_freq'
	DATA_SUB_DIR="SemEval_3way"
	mod="PretrainedModels/Hindi/${pretrainedModel}/final_model"
elif [ $LANG == "ES" ] 
then
	TASK="Sentiment_EN_ES"
	pretrainedModel='en_es_freq'
	DATA_SUB_DIR=""
	mod="PretrainedModels/Spanish/${pretrainedModel}/final_model"
elif [ $LANG == "ML" ] 
then
	TASK="Sentiment_EN_ML"
	pretrainedModel='en_ml_freq'
	DATA_SUB_DIR=""
	mod="PretrainedModels/Malayalam/${pretrainedModel}/final_model"
elif [ $LANG == "TA" ] 
then
	TASK="Sentiment_EN_TM"
	pretrainedModel='en_ta_sa_freq'
	DATA_SUB_DIR=""
	mod="PretrainedModels/Tamil/${pretrainedModel}/final_model"
else
	echo Invalid language
	exit
fi


dir=`basename "$TASK"`

if [ $dir == "Devanagari" ] || [ $dir == "Romanized" ]; then
  OUT=`dirname "$TASK"`
else
  OUT=$TASK
fi

# OUTPUT_DIR_FINE="new_switch_fine_spanishtry2"
OUTPUT_DIR_FINE="${pretrainedModel}"

echo $TASK, $mod

#for sst use 350 steps, for semeval use 100 steps, for ml finetue use 40
for seed in  32 42 52 62 72 82
do
	for epochs in 10
	do
		saveFileStart="${pretrainedModel}"

		echo Seed: $seed, Epoch: $epochs

		python3.6 $PWD/Code/utils/BertSequenceTrained.py \
		--output_dir $OUT_DIR/$OUT/$OUTPUT_DIR_FINE \
		--data_dir $DATA_DIR/$TASK/$DATA_SUB_DIR \
		--model_type $MODEL_TYPE \
		--model_name $MODEL \
		--num_train_epochs $epochs \
		--train_batch_size $BATCH_SIZE \
		--logging_steps 100 \
		--eval_batch_size $BATCH_SIZE \
		--save_steps 1 \
		--seed $seed  \
		--learning_rate 5e-5 \
		--do_train \
		--do_pred \
		--max_seq_length $MAX_SEQ \
		--gradient_accumulation_steps 1 \
		--model_loc $mod \
		--save_file_start $saveFileStart \
		# --wandb \
		# --experiment-name "test_${pretrainedModel}_seed${seed}_epoch${epochs}_testset"

	done
done