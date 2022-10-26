targetFile=${1:-'../en_hi_switch.txt'}
sourceDir=${2:-'../../../Data/MLM/withLIDtags/Hindi/combined'}

## English-Hindi
    # Experiment Switch Inverted LID
    # Replace EN with HI and HI with EN (example)
    # targetFile='../en_hi_switch_inverted.txt'
    # sourceDir='../../experiments/invertLID'

    # Standard Switch MLM
    # Use combined.txt
    # targetFile='../en_hi_switch.txt'
    # sourceDir='../../../Data/MLM/withLIDtags/Hindi/combined'

    # FreqMLM
    # targetFile='../Hindi/en_hi_freq.txt'
    # sourceDir='../../../freqMLM/data/Hindi/Hindi-freqmlm-lidtags-processed-noamb.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Hindi/en_hi_sa_baseline.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/sa_lid.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Hindi/en_hi_sa_freq.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/hindi/sa-freq-noamb.txt'


## English-Spanish
    # Baseline: masktype: all-token
    # targetFile='../en_es_baseline.txt'
    # sourceDir='../../../Data/MLM/Spanish/all_lid.txt'

    # FreqMLM
    # targetFile='../Spanish/en_es_freq.txt'
    # sourceDir='../../../freqMLM/data/Spanish/Spanish-freqmlm-lidtags-processed-noamb.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Spanish/en_es_sa_baseline.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/spanish/sa_en.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Spanish/en_es_sa_freq.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/spanish/sa-freq-processed-noamb.txt'

## English-Malayalam
    # Baseline: masktype: all-tokens
    # targetFile='../Malayalam/en_ml_baseline.txt'
    # sourceDir='../../../Data/MLM/Malayalam/all_lid.txt'

    # FreqMLM
    # targetFile='../Malayalam/en_ml_freq.txt'
    # sourceDir='../../../freqMLM/data/Malayalam/Malayalam-freqmlm-lidtags-processed-nll-noamb.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Malayalam/en_ml_sa_baseline.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/malayalam/sa_en.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Malayalam/en_ml_sa_freq.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/malayalam/sa-freq-processed-noamb.txt'

## English-Tamil
    # Baseline: masktype: all-tokens
    # targetFile='../Tamil/en_ta_baseline.txt'
    # sourceDir='../../../Data/MLM/Tamil/all_lid.txt'

    # FreqMLM
    # targetFile='../Tamil/en_ta_freq.txt'
    # sourceDir='../../../freqMLM/data/Tamil/Tamil-freqmlm-lidtags-processed-nll-noamb.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Tamil/en_ta_sa_baseline.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/tamil/sa_en.txt'

    # Experiment: Pretrain with fine tune data (SA)
    # targetFile='../Tamil/en_ta_sa_freq.txt'
    # sourceDir='../../experiments/pretrain-finetune-data/data/tamil/sa-freq-processed-noamb.txt'


python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type around-switch