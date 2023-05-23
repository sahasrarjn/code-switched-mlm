num_sent_inK=${1:-185}
# targetFile=${2:-"../../taggedData/Hindi/en_hi_baseline_l3cube_${num_sent_inK}k.txt"}
# sourceDir=${3:-"../../../Data/MLM/Hindi-l3cube/l3cube_10M_en_tagged.txt"}

targetFile="../../taggedData/Hindi/en_hi_switch_l3cube_${num_sent_inK}k.txt"
sourceDir="../../taggedData/Hindi/l3cube_cost-benefit/switch_xlmr/l3cube_985k_xlmr_tagged.txt"

num_sent=$((1000 * $num_sent_inK))

python3 preprocessMLMdata.py \
    --source $sourceDir \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --num-sent $num_sent \
    --mask-type around-switch
