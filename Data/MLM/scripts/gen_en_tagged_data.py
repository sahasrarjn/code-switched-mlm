import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='Spanish/all.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='Spanish/all_lid.txt', help="Target file")
args = parser.parse_args()

with open(args.target, 'w+') as t:
	with open(args.source, 'r') as s:
		lines = s.readlines()
		for line in tqdm(lines):
			for word in line.strip().split():
				t.write(word + '\tEN\n')
			t.write('\n')

