#!/bin/bash

: '
Generate input file for the training script

Args:
(bash $1) --target: Taget file with tokenized words and maskable tokens
(bash $2) --source: Source file with LID tags
--tokenizer-name: Tokenizer to be used
--mask-type: Selecting maskable tokens. Options: [all-tokens, around-switch]
    1. all-tokens: All the tokens are maskable. Standard MLM
    2. around-switch: Only the switch point tokens are maskable. Switch and Freq MLM
'

targetFile=${1:-'../en_hi_switch.txt'}
source=${2:-'../../../Data/LIDtagged/Hindi/pretagged.txt'}

python3 preprocessMLMdata.py \
    --source $source \
    --target $targetFile \
    --tokenizer-name bert-base-multilingual-cased \
    --mask-type all-tokens