# model name, file names, % words
import string
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
    # scores = read_scores('Code/PerplexityMasking/tmp/train_scores.txt')
    # sents = read_sents('Code/PerplexityMasking/real_CS_data/spanish_switch_sents.txt')
    # all_masks(scores, sents, 'Code/PerplexityMasking/tmp/all.txt')
    scores = read_scores('scores_filttcs_train.txt')
    sents = read_sents('../../Data/MLM/TCS/filtered_TCS_tr_realCSsize.txt')
    all_masks(scores, sents, 'mask_pscores_filttcs_token.txt')
