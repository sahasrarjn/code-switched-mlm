# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWDp
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
# MODEL=${2:-xlm-roberta-base}
# MODEL_TYPE=${3:-xlm-roberta}
OUT_DIR=${4:-$REPO/PretrainedModels/en_hi_baseline}
TRAIN_FILE=${5:-$REPO/Code/taggedData/en_hi_baseline_train.txt}
EVAL_FILE=${6:-$REPO/Code/taggedData/en_hi_baseline_eval.txt}
MLM_PROBABILITY=${7:-0.15}

# export NVIDIA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=${8:-2}
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

## L3cube Hindi
    # Baseline MLM just the 40k labelled ones
    # OUT_DIR='PretrainedModels/Hindi/l3cube40k_en_hi_baseline'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube40k_en_hi_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube40k_en_hi_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # SwitchMLM just the 40k labelled ones
    # OUT_DIR='PretrainedModels/Hindi/l3cube40k_en_hi_switch'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube40k_en_hi_switch_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube40k_en_hi_switch_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/l3cube40k_en_hi_switch.txt)

    # Baseline MLM 185k unlabelled sentences
    # OUT_DIR='PretrainedModels/Hindi/l3cube185k_en_hi_baseline'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube185k_en_hi_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube185k_en_hi_baseline_eval.txt'
    # MLM_PROBABILITY=0.15
    
    # FreqMLM 185k unlabelled sentences
    # OUT_DIR='PretrainedModels/Hindi/l3cube185k_en_hi_freq'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube185k_en_hi_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube185k_en_hi_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/l3cube185k_en_hi_freq.txt)

    # BaselineMLM 140k (unlabelled, gluecos tagged) + 44k (labelled)
    # OUT_DIR='PretrainedModels/Hindi/l3cube140k44k_en_hi_baseline'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # FreqMLM 140k44k
    # OUT_DIR='PretrainedModels/Hindi/l3cube140k44k_en_hi_freq'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/l3cube140k44k_en_hi_freq.txt)

    # SwitchMLM 140k44k
    # OUT_DIR='PretrainedModels/Hindi/l3cube140k44k_en_hi_switch'
    # TRAIN_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_switch_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/l3cube140k44k_en_hi_switch_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/l3cube140k44k_en_hi_switch.txt)

## Hindi
    # Baseline MLM
    # OUT_DIR='PretrainedModels/Hindi/en_hi_baseline'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # Standard Switch MLM
    # OUT_DIR='PretrainedModels/Hindi/en_hi_switch_xlmr'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_switch_xlmr_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_switch_xlmr_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/en_hi_switch_xlmr.txt)

    # FreqMLM
    # OUT_DIR='PretrainedModels/Hindi/en_hi_freq_xlmr'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_freq_train.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/en_hi_freq.txt)

    # Experiment: FreqMLM-Complement
    # OUT_DIR='PretrainedModels/en_hi_freq_complement'
    # TRAIN_FILE='Code/taggedData/en_hi_freq_complemen_train.txt'
    # EVAL_FILE='Code/taggedData/en_hi_freq_complemen_eval.txt'
    # MLM_PROBABILITY=0.273

    # Experiment: SwitchMLM-Complement
    # OUT_DIR='PretrainedModels/en_hi_switch_complement'
    # TRAIN_FILE='Code/taggedData/en_hi_switch_complemen_train.txt'
    # EVAL_FILE='Code/taggedData/en_hi_switch_complemen_eval.txt'
    # MLM_PROBABILITY=0.283

    # Experiment: Inverted LID SwitchMLM
    # OUT_DIR='PretrainedModels/Hindi/en_hi_switch_residbert_8_0.5'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_switch_inverted_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_switch_inverted_eval.txt'
    # MLM_PROBABILITY=0.32

    # Experiment: Maskable OTHER tokens (on top of FreqMLM)
    # OUT_DIR='PretrainedModels/en_hi_freq_maskableOTHER'
    # TRAIN_FILE='Code/taggedData/en_hi_freq_maskableOTHER_train.txt'
    # EVAL_FILE='Code/taggedData/en_hi_freq_maskableOTHER_eval.txt'
    # MLM_PROBABILITY=0.234

    # Experiment: Incorporate AMB tokens (on top of FreqMLM)
    # OUT_DIR='PretrainedModels/en_hi_freq_AMB_r2'
    # TRAIN_FILE='Code/experiments/amb-tokens/en_hi_freq_amb_train.txt'
    # EVAL_FILE='Code/experiments/amb-tokens/en_hi_freq_amb_eval.txt'
    # Output(data generation): For 181808 sentences, average fraction of english tokens is 0.24, and average number of switch points are 0.54. The average mask ratio is MASK0:0.23, MASK1:0.2, MASK2:0.03, MASK#:0.46
    # r1: (1:1.5:2) :: (0.25423728813559315 : 0.3813559322033897 : 0.5084745762711863)
    # r2: (2:1.5:1) :: (0.38 : 0.285 : 0.19)
    # --amb_tokens
    # --mask0_probability 
    # --mask1_probability 
    # --mask2_probability 

    # Experiment: Pretrain with SA finetune data
    # OUT_DIR='PretrainedModels/Hindi/en_hi_sa_baseline'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_sa_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_sa_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # Pretrain with SA finetune data (freq)
    # OUT_DIR='PretrainedModels/Hindi/en_hi_sa_freq'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_sa_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_sa_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/en_hi_sa_freq.txt)


    # Pretrain with SA finetune data (freq)
    # OUT_DIR='PretrainedModels/Hindi/en_hi_sa_freq'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_sa_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_sa_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/en_hi_sa_freq.txt)

    # Experiment: cost benefit: baseline MLM l3cube
    # OUT_DIR='PretrainedModels/Hindi/en_hi_baseline_l3cube_985k'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_baseline_l3cube_985k_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_baseline_l3cube_985k_eval.txt'
    # MLM_PROBABILITY=0.15

    # Experiment: cost benefit: switch MLM l3cube
    # OUT_DIR='PretrainedModels/Hindi/en_hi_switch_l3cube_985k'
    # TRAIN_FILE='Code/taggedData/Hindi/en_hi_switch_l3cube_985k_train.txt'
    # EVAL_FILE='Code/taggedData/Hindi/en_hi_switch_l3cube_985k_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Hindi/en_hi_switch_l3cube_985k.txt)


## Spanish
    # Baseline
    # OUT_DIR='PretrainedModels/Spanish/en_es_baseline'
    # TRAIN_FILE='Code/taggedData/Spanish/en_es_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Spanish/en_es_baseline_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Spanish/en_es_baseline.txt)

    # FreqMLM
    # OUT_DIR='PretrainedModels/Spanish/en_es_freq'
    # TRAIN_FILE='Code/taggedData/Spanish/en_es_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Spanish/en_es_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Spanish/en_es_freq.txt)

    # Switch MLM
    # OUT_DIR='PretrainedModels/Spanish/en_es_switch'
    # TRAIN_FILE='Code/taggedData/Spanish/en_es_switch_train.txt'
    # EVAL_FILE='Code/taggedData/Spanish/en_es_switch_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Spanish/en_es_switch.txt)

    # Pretrain with SA finetune data (baseline)
    # OUT_DIR='PretrainedModels/Spanish/en_es_sa_baseline'
    # TRAIN_FILE='Code/taggedData/Spanish/en_es_sa_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Spanish/en_es_sa_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # Pretrain with SA finetune data (freq)
    # OUT_DIR='PretrainedModels/Spanish/en_es_sa_freq'
    # TRAIN_FILE='Code/taggedData/Spanish/en_es_sa_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Spanish/en_es_sa_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Spanish/en_es_sa_freq.txt)

## Malayalam
    # Baseline
    # OUT_DIR='PretrainedModels/Malayalam/en_ml_baseline_residbert_9_0.5'
    # TRAIN_FILE='Code/taggedData/Malayalam/en_ml_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Malayalam/en_ml_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # FreqMLM
    OUT_DIR='PretrainedModels/Malayalam/en_ml_freq_residbert_2_0.5'
    TRAIN_FILE='Code/taggedData/Malayalam/en_ml_freq_train.txt'
    EVAL_FILE='Code/taggedData/Malayalam/en_ml_freq_eval.txt'
    MLM_PROBABILITY=$(getMLMprob Code/taggedData/Malayalam/en_ml_freq.txt)

    # Pretrain with SA finetune data (baseline)
    # OUT_DIR='PretrainedModels/Malayalam/en_ml_sa_baseline'
    # TRAIN_FILE='Code/taggedData/Malayalam/en_ml_sa_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Malayalam/en_ml_sa_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # Pretrain with SA finetune data (freq)
    # OUT_DIR='PretrainedModels/Malayalam/en_ml_sa_freq'
    # TRAIN_FILE='Code/taggedData/Malayalam/en_ml_sa_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Malayalam/en_ml_sa_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Malayalam/en_ml_sa_freq.txt)

## Tamil
    # Baseline
    # OUT_DIR='PretrainedModels/Tamil/en_ta_baseline'
    # TRAIN_FILE='Code/taggedData/Tamil/en_ta_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Tamil/en_ta_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # FreqMLM
    # OUT_DIR='PretrainedModels/Tamil/en_ta_freq'
    # TRAIN_FILE='Code/taggedData/Tamil/en_ta_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Tamil/en_ta_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Tamil/en_ta_freq.txt)

    # Pretrain with SA finetune data (baseline)
    # OUT_DIR='PretrainedModels/Tamil/en_ta_sa_baseline'
    # TRAIN_FILE='Code/taggedData/Tamil/en_ta_sa_baseline_train.txt'
    # EVAL_FILE='Code/taggedData/Tamil/en_ta_sa_baseline_eval.txt'
    # MLM_PROBABILITY=0.15

    # Pretrain with SA finetune data (freq)
    # OUT_DIR='PretrainedModels/Tamil/en_ta_sa_freq'
    # TRAIN_FILE='Code/taggedData/Tamil/en_ta_sa_freq_train.txt'
    # EVAL_FILE='Code/taggedData/Tamil/en_ta_sa_freq_eval.txt'
    # MLM_PROBABILITY=$(getMLMprob Code/taggedData/Tamil/en_ta_sa_freq.txt)


echo "Starting Pretraining With:"
echo "Train: $TRAIN_FILE"
echo "Eval: $EVAL_FILE"
echo "Output Here: $OUT_DIR"
echo "MLM prob: $MLM_PROBABILITY"

python3.6 $PWD/Code/utils/pretrain.py \
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
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --mlm_probability $MLM_PROBABILITY \
    --overwrite_output_dir \
    --residual_bert \
    --res_layer 2 \
    --res_dropout 0.5 \
    # --amb_tokens \
    # --mask0_probability 0.38 \
    # --mask1_probability 0.285 \
    # --mask2_probability 0.19

echo "Find Output Here: $OUT_DIR"

