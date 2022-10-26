from distutils.command.config import LANG_EXT
import os
import re
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', type=str, default='HI', choices=['HI', 'ES', 'ML', 'TA'], help='EN-X, Language X for CS data')
parser.add_argument('-a', '--algo', type=str, default='nll', choices=['nll', 'x-hit'], help='Masking algorithm')
args = parser.parse_args()

IGNORE_TAIL = True

lang = args.lang

if lang == 'HI':    Lang = 'Hindi'
elif lang == 'ML':  Lang = 'Malayalam'
elif lang == 'TA':  Lang = 'Tamil'
elif lang == 'ES':  Lang = 'Spanish'
else:   raise Exception('Invalid language id')



os.makedirs(f'../data/{Lang}', exist_ok=True)
sentence_file = f'../../Data/MLM/{Lang}/all_clean.txt'
processed_file = f'../data/{Lang}/{Lang}-freqmlm-lidtags-processed.txt'
processed_file_noamb = f'../data/{Lang}/{Lang}-freqmlm-lidtags-processed-noamb.txt'
output_file = f'../data/{Lang}/{Lang}-freqmlm-lidtags.txt'
EN_VOCAB = '../vocab/vocab_EN.txt'
X_VOCAB = f'../vocab/vocab_{lang}.txt'
x_aksharantar_vocab = f'../vocab/aksharantar/vocab_{lang}.txt' if lang != 'ES' else None



vocab_en = dict()
vocab_x = dict()
vocab_x_aks = set()

tail_x = 0
tail_en = 0

with open(EN_VOCAB, 'r') as env:
    for line in env.readlines():
        word, nll = line.strip().split()
        vocab_en[word] = float(nll)
        tail_en = max(float(nll), tail_en)

with open(X_VOCAB, 'r') as hiv:
    for line in hiv.readlines():
        word, nll = line.strip().split()
        vocab_x[word] = float(nll)
        tail_x = max(float(nll), tail_x)

tail = min(tail_x, tail_en)

if x_aksharantar_vocab:
    with open(x_aksharantar_vocab, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            vocab_x_aks.add(word)

sentences = []
with open(sentence_file, 'r') as sentfile:
    for line in sentfile.readlines():
        line = line.strip()
        sentences.append([word for word in line.split(' ')])

# outfile = open(output_file, 'w+')
processedfile = open(processed_file, 'w+') # just for visualizing, which word gets which tag
if args.algo == 'nll': processed_noamb = open(processed_file_noamb, 'w+')

def remove_special_chars(data):
    special_char = re.compile(r'[@_!#$%.`^&,*()<>?/\'\|"“}{~:-]')
    return re.sub(special_char, ' ', data)


def nll_mask(word):
    word_nll_en = vocab_en.get(word, -1)
    word_nll_x = vocab_x.get(word, -1)

    if IGNORE_TAIL:
        if word_nll_en > tail: word_nll_en = -1
        if word_nll_x > tail: word_nll_x = -1

    mask = None
    if word_nll_en == -1 and word_nll_x == -1:
        mask = "OTHER"
    elif word_nll_x == -1 and word_nll_en != -1:
        mask = "EN" # this case seems a bit flimsy, what if its extremely rare in hindi and doesnt occur in english??
    elif word_nll_en == -1 and word_nll_x != -1:
        mask = args.lang
    elif word_nll_x + np.log(10) < word_nll_en:
        mask = args.lang
    elif word_nll_en + np.log(10) < word_nll_x:
        mask = "EN"
    elif np.abs(word_nll_en - word_nll_x) < np.log(10):
        if word_nll_en < word_nll_x:
            mask = "AMB-EN"
        else:
            mask = f"AMB-{args.lang}"
    return mask

def x_present_mask(word):
    '''If word is present in the X's vocab (Aksharantar), mask it as X or otherwise mask as EN'''
    if x_aksharantar_vocab is None:
        raise Exception(f'Incorrect language pair for {args.algo} algorithm, choose from HI, ML or TA')

    mask = None
    if word in vocab_x_aks:
        return args.lang
    
    word_nll_en = vocab_en.get(word, -1)
    if word_nll_en == -1:
        mask = "OTHER"
    else:
        mask = "EN"
    
    return mask


for sentence in tqdm(sentences):
    for word in sentence:
        if word == '':
            continue
        word = word.lower()
        word = remove_special_chars(word)

        if args.algo == 'nll':
            mask = nll_mask(word)
        elif args.algo == 'x-hit':
            mask = x_present_mask(word)
        else:
            raise NotImplementedError

        # outfile.write(f'{mask} ')
        if args.algo == 'nll': processed_noamb.write(word + '\t' + f'{mask if mask[:3] != "AMB" else mask[-2:]}' + '\n')
        processedfile.write(word + '\t' + mask + '\n')
    # outfile.write('\n')
    if args.algo == 'nll': processed_noamb.write('\n')
    processedfile.write('\n')

# outfile.close()
if args.algo == 'nll': processed_noamb.close()
processedfile.close()
