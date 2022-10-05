# model name, file names, % words
import string
import argparse
from statistics import mean, stdev
import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

np.random.seed(100)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def read_scores(filename):
    scores = []
    with open(filename, 'r') as f:
        scores = [np.array(l.split(' ')).astype(np.float) for l in f.readlines()]
    return scores

def read_sents(filename):
    sents = []
    with open(filename, 'r') as f:
        sents = [l.strip() for l in f.readlines()]
    return sents

def stable_expo(x):
    z = x - max(x)
    numerator = np.power(2, z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def create_masks(sent_score, sent, tokenMap):
    # print(sent_score)
    smax = stable_expo(sent_score)
    # print(smax)
    n = smax.shape[0]
    masks = []
    msk = np.random.choice(np.arange(n), round(0.4*n), p=smax, replace=False)
    # print(len(msk)/n)
    # print(msk)
    xx = sent.split(' ')
    if(len(xx)!=n):
        print(sent)
    for i in range(n):
        word = xx[i]
        fill = np.full(len(tokenMap[word]), "MASK").tolist() if i in msk else np.full(len(tokenMap[word]), "NOMASK").tolist()
        masks.extend(fill)
    # print(masks.count("MASK")/n)
    return masks, masks.count("MASK")/len(masks)

def all_masks(scores, sentences, outfile):
    print(outfile)
    avg_msk = 0
    with open(outfile, 'w') as f:
        for s, l in tqdm(zip(sentences, scores)):
            tokenMap = {}
            for word in s.split(' '):
                tokenized = tokenizer.tokenize(word)
                tokenMap[word] = tokenized
            masks, am = create_masks(l, s, tokenMap)
            avg_msk = avg_msk+ am
            f.write(s+'\t'+' '.join(masks)+'\n')
    print(avg_msk/len(scores))

if __name__=='__main__':
    desc = "Use correlation score to identify maskable tokens"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("-s", "--score_file", required=True, help="File with perplexity scores")
    parser.add_argument("-i", "--input_file", required=True, help="Input file with corresponding sentences")
    parser.add_argument("-o", "--output_file", required=True, help="Output file")
    args = parser.parse_args()

    scores = read_scores(args.score_file)
    sents = read_sents(args.input_file)
    all_masks(scores, sents, args.output_file)
