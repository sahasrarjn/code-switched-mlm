# Experiment maskable OTHER tokens:
#   Instead of ignoring the OTHER labeled tokens from freqMLM, add them to the set of tokens that are maskable that currently consists of only switch points. 

targetFile=${1:-'../../taggedData/en_hi_freq_maskableOTHER.txt'}
sourceDir=${2:-'../../../freqMLM/data/CSdata/emoji/combined-freqmlm-lidtags-processed-noabm.txt'}

python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type around-switch \
