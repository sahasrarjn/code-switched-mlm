'''
Generate input file with all words OTHER tags

Input args:
    -s, --source: Source file
    -t, --target: Target file
'''

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../Hindi/L3Cube-HingLID/all_original.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='../Hindi/L3Cube-HingLID/all.txt', help="Target file")
args = parser.parse_args()

with open(args.target, 'w+') as fo:
    with open(args.source, 'r') as fi:
        lines = fi.readlines()

        for line in tqdm(lines):
            word = line.split('\t')[0]
            if word.isnumeric():
                line = word + '\tOTHER\n'
            fo.write(line)
