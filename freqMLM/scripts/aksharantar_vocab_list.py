'''
Generate vocabulary file from Aksharantar vocab dataset

Input args:
    -s, --source:   Source dir
    -d, --dest:     Output vocab file
'''


import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source_dir', type=str, default='../data/Aksharantar/hin', help='Source directory')
parser.add_argument('-d', '--dest', type=str, default='../vocab/aksharantar/vocab_HI.txt')
args = parser.parse_args()

vocab = set([])
dakshinaCount = 0

for filename in os.listdir(args.source_dir):
    filename = os.path.join(args.source_dir, filename)
    print(f'Reading file {filename}')
    with open(filename, 'r') as f:
        data = json.load(f)
        for word in tqdm(data):
            if word['source'] != 'AK-NEF':
                vocab.add(word['english word'])
            else:
                # vocab.add(word['english word'])
                dakshinaCount += 1

vocab = list(vocab)
vocab = sorted(vocab)

print(f'Ignored {dakshinaCount} words from AK-NEF')

print(f'Writing to file: {args.dest}')
with open(args.dest, 'w+') as f:
    for v in vocab:
        f.write(v + '\n')

