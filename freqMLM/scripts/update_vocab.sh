# Options: HI, ES, TA, ML
LID=${1:-ML} 
lid=$(echo "$LID" | tr '[:upper:]' '[:lower:]')

if [ "$LID" = "HI" ]; then
    echo $LID
    # python3 freqMLM_generate_vocab.py ## Not working, update
else
    echo $LID
    python3 gen_vocab_from_sentences.py -i ../data/Samanantar/en-${lid}/train.${lid}.romanized -o ../vocab/vocab_${LID}.txt -p ../vocab/vocab_${LID}.png 
fi

python3 gen_freqmlm_tags.py -l ${LID}