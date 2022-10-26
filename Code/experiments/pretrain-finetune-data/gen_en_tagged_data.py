import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='data/sa.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='data/sa_lid.txt', help="Target file")
args = parser.parse_args()

with open(args.target, 'w+') as t:
	with open(args.source, 'r') as s:
		lines = s.readlines()
		for line in lines:
			for word in line.strip().split():
				t.write(word + '\tEN\n')
			t.write('\n')

