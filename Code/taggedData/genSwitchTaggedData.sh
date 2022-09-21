targetFile=${1:-'../en_hi_freq_woemoji.txt'}
sourceDir=${2:-'../../Data/MLM/withLIDtags/Hindi/woemoji_freq'}

python3 utils/preprocessMLMdata.py \
    --source-dir $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type around-switch \
    # --debug
    # --source-dir $PWD/Data/MLM/freqMLMwithLIDtags \
