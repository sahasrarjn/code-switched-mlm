python3 freqMLM_generate_vocab.py 
echo VOCAB GENERATED !!!
python3 gen_freqmlm_tags.py
echo TAGS GENERATED !!!
python3 gen_true_pred_table.py
echo CONFUSION MATRIX GENERATED !!!