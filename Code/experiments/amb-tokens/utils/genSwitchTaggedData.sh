## English-Hindi
# Experiment: Use AMB tokens

targetFile=${1:-'../en_hi_freq_amb.txt'}
sourceDir=${2:-'../../../../freqMLM/data/CSdata/emoji/combined-freqmlm-lidtags-processed.txt'}
# Output: For 181808 sentences, average fraction of english tokens is 0.24, and average number of switch points are 0.54. The average mask ratio is MASK0:0.23, MASK1:0.2, MASK2:0.03, MASK#:0.46


# targetFile=${2:-'../output.txt'}
# sourceDir=${2:-'../input.txt'}

python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type around-switch-amb