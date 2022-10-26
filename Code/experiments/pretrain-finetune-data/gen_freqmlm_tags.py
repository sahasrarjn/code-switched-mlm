import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', type=str, default='HI', choices=['HI', 'ES', 'ML', 'TA'], help='EN-X, Language X for CS data')
args = parser.parse_args()

EN_VOCAB = '../../../freqMLM/vocab/vocab_EN.txt'

if args.lang == 'HI':
    os.makedirs('data/hindi/', exist_ok=True)
    sentence_file = 'data/hindi/sa.txt'
    processed_file = 'data/hindi/sa-freq-processed.txt'
    output_file = 'data/hindi/sa-freq.txt'
    processed_file_noamb = 'data/hindi/sa-freq-noamb.txt'
    X_VOCAB = '../../../freqMLM/vocab/vocab_HI.txt'
elif args.lang == 'ML':
    os.makedirs('data/malayalam/', exist_ok=True)
    sentence_file = 'data/malayalam/sa.txt'
    processed_file = 'data/malayalam/sa-freq-processed.txt'
    output_file = 'data/malayalam/sa-freq.txt'
    processed_file_noamb = 'data/malayalam/sa-freq-processed-noamb.txt'
    X_VOCAB = '../../../freqMLM/vocab/vocab_ML.txt'
elif args.lang == 'TA':
    os.makedirs('data/tamil/', exist_ok=True)
    sentence_file = 'data/tamil/sa.txt'
    processed_file = 'data/tamil/sa-freq-processed.txt'
    output_file = 'data/tamil/sa-freq.txt'
    processed_file_noamb = 'data/tamil/sa-freq-processed-noamb.txt'
    X_VOCAB = '../../../freqMLM/vocab/vocab_TA.txt'
elif args.lang == 'ES':
    os.makedirs('data/spanish/', exist_ok=True)
    sentence_file = 'data/spanish/sa.txt'
    processed_file = 'data/spanish/sa-freq-processed.txt'
    output_file = 'data/spanish/sa-freq.txt'
    processed_file_noamb = 'data/spanish/sa-freq-processed-noamb.txt'
    X_VOCAB = '../../../freqMLM/vocab/vocab_ES.txt'
else:
    raise Exception('Invalid language id')


vocab_en = dict()
vocab_x = dict()
with open(EN_VOCAB, 'r') as env:
    for line in env.readlines():
        word, nll = line.strip().split()
        vocab_en[word] = float(nll)

with open(X_VOCAB, 'r') as hiv:
    for line in hiv.readlines():
        word, nll = line.strip().split()
        vocab_x[word] = float(nll)

sentences = []
with open(sentence_file, 'r') as sentfile:
    for line in sentfile.readlines():
        line = line.strip()
        sentences.append([word for word in line.split(' ')])

outfile = open(output_file, 'w+')
processedfile = open(processed_file, 'w+') # just for visualizing, which word gets which tag
processed_noamb = open(processed_file_noamb, 'w+')

for sentence in tqdm(sentences):
    for word in sentence:
        if word == '':
            continue
        word = word.lower()
        word_nll_en = vocab_en.get(word, -1)
        word_nll_hi = vocab_x.get(word, -1)
        mask = None
        if word_nll_en == -1 and word_nll_hi == -1:
            mask = "OTHER"
        elif word_nll_hi == -1 and word_nll_en != -1:
            mask = "EN" # this case seems a bit flimsy, what if its extremely rare in hindi and doesnt occur in english??
        elif word_nll_en == -1 and word_nll_hi != -1:
            mask = args.lang
        elif word_nll_hi + np.log(10) < word_nll_en:
            mask = args.lang
        elif word_nll_en + np.log(10) < word_nll_hi:
            mask = "EN"
        elif np.abs(word_nll_en - word_nll_hi) < np.log(10):
            if word_nll_en < word_nll_hi:
                mask = "AMB-EN"
            else:
                mask = f"AMB-{args.lang}"
        outfile.write(f'{mask} ')
        # outfile_noamb.write(f'{mask if mask[:3] != "AMB" else mask[-2:]} ')
        processed_noamb.write(word + '\t' + f'{mask if mask[:3] != "AMB" else mask[-2:]}' + '\n')
        processedfile.write(word + '\t' + mask + '\n')
    outfile.write('\n')
    processed_noamb.write('\n')
    processedfile.write('\n')

outfile.close()
processed_noamb.close()
processedfile.close()
