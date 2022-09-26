import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='../../Data/MLM/withLIDtags/Hindi/combined/combined.txt', help='Input LID file switch LID tags')
parser.add_argument('-o', '--outfile', type=str, default='invertedLID_EN_HI.txt', help='Output file for data with inverted LID tags')
parser.add_argument('-l', '--lang', type=str, default="Hindi", choices=["Hindi", "Spanish"], help="Language CS with English")
args = parser.parse_args()

input_file = args.file
output_file = args.outfile

lang_ids = ['EN']
if args.lang == 'Hindi':
    lang_ids.append('HI')
elif args.lang == 'Spanish':
    lang_ids.append('ES')
else:
    raise Exception("Invalid input langauge")

try:
    os.path.isfile(input_file)
    os.path.isfile(output_file)
except:
    raise Exception("Invalid file path")

def invert_lid(tag):
    if tag == 'OTHER':
        return tag
    elif tag == lang_ids[0]:
        return lang_ids[1]
    elif tag == lang_ids[1]:
        return lang_ids[0]
    else:
        raise Exception(f"Invalid tag {tag}")

with open(input_file, 'r') as fi:
    lines = fi.readlines()
    with open(output_file, 'w+') as fo:
        for line in tqdm(lines):
            if line.strip() == '':
                fo.write(line)
                continue
            
            word, lid = line.rstrip().split('\t')
            newlid = invert_lid(lid)
            fo.write(word + '\t' + newlid + '\n')