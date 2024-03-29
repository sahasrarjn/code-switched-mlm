'''
Generate frequence MLM tags using the input vocab file and input code switched sentences

Input args:
> -l, --lang: Matrix lanauge for the code switched text data. Options: [HI, ES, ML, TA]
> -a, --algo: Algorithm for Freq MLM. Options: [x-hit, nll]


This script generated two file, defined below as processed_file and processed_file_noamb. As name suggest the later file no not contain any ambiguous lid tags and mark them as either the matrix or the embedded langauge
'''

from distutils.command.config import LANG_EXT
import os
import re
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', type=str, default='HI', help='X-EN, Language X for CS data')
parser.add_argument('-a', '--algo', type=str, default='nll', choices=['nll', 'x-hit'], help='Masking algorithm')
args = parser.parse_args()

IGNORE_TAIL = True

lang = args.lang

if lang == 'HI':    Lang = 'Hindi'
elif lang == 'ML':  Lang = 'Malayalam'
elif lang == 'TA':  Lang = 'Tamil'
elif lang == 'ES':  Lang = 'Spanish'
elif lang == "HI-L3CUBE": lang, Lang = 'HI', 'Hindi-l3cube'
elif lang == "HI-L3CUBE140k44k": lang, Lang = 'HI', 'Hindi-l3cube140k44k'
else:   raise Exception('Invalid language id')



os.makedirs(f'../data/{Lang}', exist_ok=True)

# Google no swear words
google_10k_words = '../data/google-10000-english-no-swears.txt'

# Input sentence file
sentence_file = f'../../Data/MLM/{Lang}/all_clean.txt'


# Output LID tagged files
processed_file = f'../../Data/LIDtagged/{Lang}/{Lang}-freqmlm-lidtags-processed-{args.algo}.txt'
processed_file_noamb = f'../../Data/LIDtagged/{Lang}/{Lang}-freqmlm-lidtags-processed-{args.algo}-noamb.txt'


# Input vocabulary files
EN_VOCAB = '../vocab/vocab_EN.txt'
X_VOCAB = f'../vocab/vocab_{lang}.txt'
x_aksharantar_vocab = f'../vocab/aksharantar/vocab_{lang}.txt' if (lang != 'ES' and args.algo == 'x-hit') else None



vocab_en = dict()
vocab_x = dict()

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
    vocab_x_aks = set()
    common_eng_words = set()

    with open(google_10k_words, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            common_eng_words.add(word)
    
    with open(x_aksharantar_vocab, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            if word not in common_eng_words:
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
        mask = lang
    elif word_nll_x + np.log(10) < word_nll_en:
        mask = lang
    elif word_nll_en + np.log(10) < word_nll_x:
        mask = "EN"
    elif np.abs(word_nll_en - word_nll_x) < np.log(10):
        if word_nll_en < word_nll_x:
            mask = "AMB-EN"
        else:
            mask = f"AMB-{lang}"
    return mask

def x_present_mask(word):
    '''If word is present in the X's vocab (Aksharantar), mask it as X or otherwise mask as EN'''
    if x_aksharantar_vocab is None:
        raise Exception(f'Incorrect language pair for {args.algo} algorithm, choose from HI, ML or TA')

    mask = None
    if word in vocab_x_aks:
        return lang
    
    word_nll_en = vocab_en.get(word, -1)
    if word_nll_en == -1:
        if word.isnumeric():
            # print(word)
            mask = "OTHER"
        else:
            mask = lang
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
            if lang == 'ES': raise Exception("Aksharantar vocab not present for Spanish")
            mask = x_present_mask(word)
        else:
            raise NotImplementedError

        if args.algo == 'nll': processed_noamb.write(word + '\t' + f'{mask if mask[:3] != "AMB" else mask[-2:]}' + '\n')
        processedfile.write(word + '\t' + mask + '\n')
    if args.algo == 'nll': processed_noamb.write('\n')
    processedfile.write('\n')

if args.algo == 'nll': processed_noamb.close()
processedfile.close()
