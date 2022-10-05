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
# targetFile='../en_es_baseline.txt'
# sourceDir='../../../Data/MLM/withLIDtags/Spanish/final'


python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type all-tokens \