#!/usr/bin/python3

'''
Split input file in train and eval set for pretraining

Input args:
-f --file: Input file with maskable tokens
--frac: Split fraction: set to 90% train, 10% eval
--seed: Random seed, set to 0
'''

import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='../en_hi_switch.txt', help='Input file to split into train, eval files')
parser.add_argument('--frac', type=float, default=0.1, help='Split fraction for train/eval')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)

input_file = args.file
try:
    assert(input_file[-4:] == '.txt')
except:
    raise Exception("Invalid file path, must be a txt file")

basepath = input_file.rstrip('.txt')
train_file = basepath + '_train.txt'
eval_file = basepath + '_eval.txt'

with open(input_file, 'r') as f:
    lines = f.readlines()
    numTotal = len(lines)
    numEval = int(args.frac * numTotal)
    evalIdx = random.sample(range(numTotal), numEval)

    print(f"Split: {train_file} ({numTotal-numEval}), {eval_file} ({numEval})")
    with open(train_file, 'w+') as tf:
        with open(eval_file, 'w+') as ef:
            for i, line in enumerate(tqdm(lines)):
                if i in evalIdx:
                    ef.write(line)
                else:
                    tf.write(line)