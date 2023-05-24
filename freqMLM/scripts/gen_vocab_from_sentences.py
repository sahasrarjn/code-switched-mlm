'''
This file generated a vocabulary with negative log likelihood counts for each words occuring in the dataset.

Input:
-i, --input_file: Input dataset file with code switched sentences
-o, --output_file: Output file for vocabulary with NLL scores
-p, --output_plotfile: Output plot files

** We already share the vocabulary we generated for the datasets we used. 
'''

import re
import csv
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, default='../data/Samanantar/en-ta/train.ta.romanized', help='Sentences data file')
parser.add_argument('-o', '--output_file', type=str, default='../vocab/vocab_TA.txt', help='Output vocab frequency file')
parser.add_argument('-p', '--output_plotfile', type=str, default='../vocab/vocab_TA.png', help='Output vocab frequency plot file')
args = parser.parse_args()


vocab = defaultdict(int)
totalCount = 0

with open(args.input_file, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        words = re.findall('[a-zA-Z]+', line)
        for word in words:
            if word.isnumeric():
                print("#"*100)
                print(word, " : ANOMALY")
                print("#"*100)
                continue
            word = word.lower()
            vocab[word] += 1
            totalCount += 1

for _, word in enumerate(vocab):
    vocab[word] = -np.log(vocab[word]) + np.log(totalCount)

with open(args.output_file, 'w+') as of:
    for word, nll in sorted(vocab.items(), key=lambda kv: kv[1]):
        of.write(word + ' ' + str(nll) + '\n')

distri = [v for _, v in vocab.items()]
distri = sorted(distri)
plt.hist(distri)
plt.ylabel('Count')
plt.xlabel('Frequency (-ve log likelihood)')
plt.savefig(args.output_plotfile)
plt.clf()