'''
Generate input file with all words EN or OTHER tags

Input args:
    -s, --source: Source file
    -t, --target: Target file
'''


# TAG = 'OTHER'
TAG = 'EN'


import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../Hindi/all_clean.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='../../LIDtagged/Hindi/allENtags.txt', help="Target file")
args = parser.parse_args()

with open(args.target, 'w+') as t:
	with open(args.source, 'r') as s:
		lines = s.readlines()
		for line in tqdm(lines):
			for word in line.strip().split():
				t.write(word + f'\t{TAG}\n')
			t.write('\n')

