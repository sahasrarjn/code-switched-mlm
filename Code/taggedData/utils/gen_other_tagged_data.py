import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../Hindi/en_hi_baseline_l3cube_985k.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='../Hindi/switch/en_hi_baseline_l3cube_985k.txt', help="Target file")
args = parser.parse_args()

with open(args.target, 'w+') as t:
	with open(args.source, 'r') as s:
		lines = s.readlines()
		for line in tqdm(lines):
			line = line.split('\t')[0]
			for word in line.strip().split():
				t.write(word + '\tOTHER\n')
			t.write('\n')

