targetFile=${1:-'../en_hi_switch.txt'}
sourceDir=${2:-'../../../Data/MLM/withLIDtags/Hindi/combined'}

## L3cube English-Hindi
    # Experiment Standard Switch MLM (just the switch data of l3cube)
    # targetFile='../Hindi/l3cube_en_hi_switch.txt'
    # sourceDir='../../../Data/MLM/Hindi/L3Cube-HingLID/processed/l3cube_44k_og_tagged.txt'

    # Experiment Frequency MLM on 185k sentences sampled from L3CUBE-unlabelled 52 million sents
    # targetFile='../Hindi/l3cube185k_en_hi_freq.txt'
    # sourceDir='../../../freqMLM/data/Hindi-l3cube/Hindi-l3cube-freqmlm-lidtags-processed-nll-noamb.txt'

    # Experiment Baseline MLM on 185k sentences sampled from L3CUBE-unlabelled 52 million sents
    # Baseline: masktype: all-tokens
    # targetFile='../Hindi/l3cube185k_en_hi_baseline.txt'
    # sourceDir='../../../freqMLM/data/Hindi-l3cube/Hindi-l3cube-freqmlm-lidtags-processed-nll-noamb.txt'

    # Experiment Switch MLM on 140k (unlabelled, gluecos tagged) + 44k (labelled) sentences sampled from L3CUBE
    # targetFile='../Hindi/l3cube140k44k_en_hi_switch.txt'
    # sourceDir='../../../Data/MLM/Hindi/L3Cube-HingLID/processed/l3cube_140k_44k_tagged.txt'

    # Experiment Frequency MLM on 140k (unlabelled) + 44k (labelled) sentences sampled from L3CUBE
    # targetFile='../Hindi/l3cube140k44k_en_hi_freq.txt'
    # sourceDir='../../../freqMLM/data/Hindi-l3cube140k44k/Hindi-l3cube140k44k-freqmlm-lidtags-processed-nll-noamb.txt'

    # Experiment Baseline MLM on 140k (unlabelled) + 44k (labelled) sentences sampled from L3CUBE
    # Baseline: masktype: all-tokens
    # targetFile='../Hindi/l3cube140k44k_en_hi_baseline.txt'
    # sourceDir='../../../Data/MLM/Hindi/L3Cube-HingLID/processed/l3cube_140k_44k_tagged.txt'


## English-Hindi
    # Baseline: masktype: all-token
    # targetFile='../Hindi/en_hi_baseline.txt'
    # sourceDir='../../../Data/MLM/withLIDtags/Hindi/combined'
    
    # Experiment Switch Inverted LID
    # Replace EN with HI and HI with EN (example)
    targetFile='../en_hi_switch_inverted.txt'
    sourceDir='../../experiments/invertLID'

    # Standard Switch MLM
    # Use combined.txt
    # targetFile='../Hindi/en_hi_switch.txt'
    # sourceDir='../../../Data/MLM/withLIDtags/Hindi/combined'

    # FreqMLM
    # targetFile='../Hindi/en_hi_freq.txt'
    # sourceDir='../../../freqMLM/data/Hindi/Hindi-freqmlm-lidtags-processed-nll.txt'

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

    # Switch MLM
    # targetFile='../Spanish/en_es_switch.txt'
    # sourceDir='../../../Data/MLM/Spanish/all_switch_lid.txt'

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
    # sourceDir='../../../freqMLM/data/Malayalam/Malayalam-freqmlm-lidtags-processed.txt'

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