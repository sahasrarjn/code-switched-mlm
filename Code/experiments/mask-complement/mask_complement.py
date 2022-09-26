import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='../../taggedData/en_hi_freq.txt', help='Take in mask file to complement')
parser.add_argument('-o', '--outfile', type=str, default='../../taggedData/en_hi_freq_complement.txt', help='Output file with the complemented MASKs')
args = parser.parse_args()

def mask_comp(mask_val):
    return 'MASK' if mask_val == 'NOMASK' else 'NOMASK'

outfile = open(args.outfile, 'w+')
count = 0
maskRatio = 0
with open(args.file, 'r') as fp:
    for line in tqdm(fp.readlines(), desc="Lines"):
        count += 1
        line = line.strip()
        og_sent, tags = line.split('\t')
        new_tags_list = list(map(mask_comp, tags.split()))
        maskRatio += new_tags_list.count('MASK')/len(new_tags_list)
        outfile.write(og_sent + '\t' + ' '.join(new_tags_list) + '\n')
outfile.close()

print(f"The average mask ratio is {np.round(maskRatio/count, 2)}")