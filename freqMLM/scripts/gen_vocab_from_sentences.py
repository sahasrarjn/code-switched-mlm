import re
import csv
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, default='../data/Samanantar/en-ml/train.ml.romanized', help='Sentences data file')
parser.add_argument('-o', '--output_file', type=str, default='../vocab/vocab_ML.txt', help='Output vocab frequency file')
parser.add_argument('-p', '--output_plotfile', type=str, default='../vocab/vocab_ML.png', help='Output vocab frequency plot file')
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