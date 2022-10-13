targetFile=${1:-'../en_hi_switch.txt'}
sourceDir=${2:-'../../../Data/MLM/withLIDtags/Hindi/combined'}

## English-Hindi
    # Experiment Switch Inverted LID
    # Replace EN with HI and HI with EN (example)
    # $1 = '../en_hi_switch_inverted.txt'
    # $2 = '../../experiments/invertLID'

    # Standard Switch MLM
    # Use combined.txt
    # $1 = '../en_hi_switch.txt'
    # $2 = '../../../Data/MLM/withLIDtags/Hindi/combined'

    # FreqMLM
    # $1 = '../en_hi_freq.txt'
    # $2 = '../../../freqMLM/data/CSdata/emoji/combined-freqmlm-lidtags-processed-noabm.txt'


## English-Spanish
    # Baseline
    # targetFile='../en_es_baseline.txt'
    # sourceDir='../../../Data/MLM/withLIDtags/Spanish/final'

    # FreqMLM
    # targetFile='../Spanish/en_es_freq.txt'
    # sourceDir='../../../freqMLM/data/Spanish/Spanish-freqmlm-lidtags-processed-noamb.txt'

## English-Malayalam
    # Baseline: masktype: all-tokens
    targetFile='../Malayalam/en_ml_baseline.txt'
    sourceDir='../../../Data/MLM/Malayalam/all_lid.txt'

    # FreqMLM
    # targetFile='../Malayalam/en_ml_freq.txt'
    # sourceDir='../../../freqMLM/data/Malayalam/Malayalam-freqmlm-lidtags-processed-noamb.txt'

## English-Tamil
    # Baseline: masktype: all-tokens
    # targetFile='../Tamil/en_ta_baseline.txt'
    # sourceDir='../../../Data/MLM/Tamil/all_lid.txt'

    # FreqMLM
    # targetFile='../Tamil/en_ta_freq.txt'
    # sourceDir='../../../freqMLM/data/Tamil/Tamil-freqmlm-lidtags-processed-noamb.txt'


python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type all-tokens