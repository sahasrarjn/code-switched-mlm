import random
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../../../Data/Processed_Data/Sentiment_EN_HI/Romanized/SemEval_3way/all.txt', help='Source file')
parser.add_argument('-t', '--target', type=str, default='../../taggedData/Hindi/en_hi_sa_fine_baseline.txt', help='Target file')
args = parser.parse_args()


with open(args.source, 'r') as s:
    lines = s.readlines()
    lines = [l.split('\t')[0] for l in lines]
        
with open(args.target, 'w+') as t:
    for line in lines:
        t.write(line + '\n')