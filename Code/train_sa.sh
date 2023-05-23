# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
LANG=$1
TASK=${2:-Sentiment_EN_HI/Romanized}

# For BERT
MODEL=${3:-bert-base-multilingual-cased}
MODEL_TYPE=${4:-bert}


# For XLMR
# MODEL=${3:-xlm-roberta-base}
# MODEL_TYPE=${4:-xlm-roberta}



DATA_DIR=${5:-"$REPO/Data/Processed_Data"}
OUT_DIR=${6:-"$REPO/Results"}
pretrainedModel=${6:-"en_hi_baseline"}
export CUDA_VISIBLE_DEVICES=${7:-2}
# export CUDA_LAUNCH_BLOCKING=1

MAX_SEQ=256


# Initial hyper params (mBERT)
BATCH_SIZE=8
LEARNING_RATE=5e-5
GRAD_ACC=1

# Optimal hyperparameter for XLMR (baseline)
# BATCH_SIZE=16
# LEARNING_RATE=5e-6
# GRAD_ACC=4

# Optimal hyperparameter for XLMR (switch)
# BATCH_SIZE=32
# LEARNING_RATE=5e-6
# GRAD_ACC=1



# baseModel is set true when base XLMR/mBERT models are used and false if custom MLM pretrained models are used
baseModel=false


if [ "$LANG" == "HI" ] ;
then
	TASK="Sentiment_EN_HI/Romanized"
	if [ $baseModel == false ]; then
		# MODEL_TYPE='residual-bert_2_0.5'
		echo Model Type: $MODEL_TYPE
	fi
	pretrainedModel='en_hi_baseline_xlmr'
	DATA_SUB_DIR="SemEval_3way"
	mod="PretrainedModels/Hindi/${pretrainedModel}/final_model"
elif [ "$LANG" == "ES" ] ; 
then
	TASK="Sentiment_EN_ES"
	pretrainedModel='en_es_baseline_xlmr'
	DATA_SUB_DIR=""
	mod="PretrainedModels/Spanish/${pretrainedModel}/final_model"
elif [ "$LANG" == "ML" ] ; 
then
	TASK="Sentiment_EN_ML"
	if [ $baseModel == false ]; then
		MODEL_TYPE='residual-bert_2_0.5'
		echo Model Type: $MODEL_TYPE
	fi
	pretrainedModel='en_ml_freq_residbert_2_0.5_aux'
	DATA_SUB_DIR=""
	mod="PretrainedModels/Malayalam/${pretrainedModel}/final_model"
elif [ "$LANG" == "TA" ] ;
then
	TASK="Sentiment_EN_TM"
	if [ $baseModel == false ]; then
		# MODEL_TYPE='residual-bert_10_0.5'
		echo Model Type: $MODEL_TYPE
	fi
	pretrainedModel='en_ta_freq_xlmr'
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


if [ $baseModel == true ]; then
	pretrainedModel=$MODEL_TYPE
	echo $TASK, $pretrainedModel
else
	echo $TASK, $mod
fi

#for sst use 350 steps, for semeval use 100 steps, for ml finetue use 40
for seed in 32 42 52 62 72 82 92 102 112 122 132 142
# for seed in  32 42 52 62 72 82
do
	for epochs in 10
	do
		saveFileStart=$pretrainedModel
		OUTPUT_DIR_FINE="${pretrainedModel}"
		echo Seed: $seed, Epoch: $epochs

		if [ $baseModel == true ]; then
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
			--learning_rate $LEARNING_RATE \
			--do_train \
			--do_pred \
			--max_seq_length $MAX_SEQ \
			--gradient_accumulation_steps $GRAD_ACC \
			--save_file_start $saveFileStart
			# --wandb
			# --experiment-name "test_${pretrainedModel}_seed${seed}_epoch${epochs}_testset"

		else
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
			--learning_rate $LEARNING_RATE \
			--do_train \
			--do_pred \
			--max_seq_length $MAX_SEQ \
			--gradient_accumulation_steps $GRAD_ACC \
			--model_loc $mod \
			--save_file_start $saveFileStart
			# --wandb
			# --experiment-name "test_${pretrainedModel}_seed${seed}_epoch${epochs}_testset"
		fi
	done
done