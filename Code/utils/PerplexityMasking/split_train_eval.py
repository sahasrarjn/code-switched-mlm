import os
import random
from tqdm import tqdm

eval_file = 'Code/PerplexityMasking/tmp/eval.txt'
train_file = 'Code/PerplexityMasking/tmp/train.txt'

random.seed(0)

total_num = 33283 
eval_num =  3000
eval = random.sample(range(total_num), eval_num)

f1 = open(train_file, 'w')
f2 = open(eval_file, 'w')

i=0
with open('Code/PerplexityMasking/tmp/all.txt', 'r') as f:
    for l in f:
        if i in eval:
            f2.write(l)
            i+=1
        else:
            f1.write(l)
            i+=1

