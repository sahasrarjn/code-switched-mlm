function getMLMprob {
    file=$1
    mask_cnt=$(grep -o -i MASK ${file} | wc -l)
    nomask_cnt=$(grep -o -i NOMASK ${file} | wc -l)
    total_tokens=$(python3 -c "print($mask_cnt + $nomask_cnt)")
    echo $(python3 -c "print( 0.15 * $total_tokens / $mask_cnt)" )
}


dsize=(
    "185k"
    "285k"
    "385k"
    "485k"
    "585k"
    "785k"
    "985k"
)

for ds in ${dsize[@]}; do
    file="en_hi_switch_l3cube_${ds}"
    echo $file
    python3 Code/taggedData/utils/split_train_eval.py -f "Code/taggedData/Hindi/${file}.txt"
    MLM_PROBABILITY=$(getMLMprob "Code/taggedData/Hindi/${file}.txt")
    bash Code/pretrain.sh SA_EN_HI bert-base-multilingual-cased bert "PretrainedModels/Hindi/${file}_xlmr_tag" "Code/taggedData/Hindi/${file}_train.txt" "Code/taggedData/Hindi/${file}_eval.txt" $MLM_PROBABILITY 3 &> "logs/pretrain/hindi/switch_l3cube_${ds}_xlmr_tag.log"
    bash Code/train_sa.sh HI Sentiment_EN_HI/Romanized bert-base-multilingual-cased bert Data/Processed_Data Results "${file}_xlmr_tag" 3 &> "logs/train_sa/hindi/switch_l3cube_${ds}_xlmr_tag.log"
done
