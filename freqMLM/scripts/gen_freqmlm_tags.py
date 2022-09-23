import numpy as np

sentence_file = '../data/CSdata/emoji/combined.txt'
processed_file = '../data/CSdata/emoji/combined-freqmlm-lidtags-processed.txt'
output_file = '../data/CSdata/emoji/combined-freqmlm-lidtags.txt'
processed_file_noamb = '../data/CSdata/emoji/combined-freqmlm-lidtags-noamb.txt'

EN_VOCAB = 'out/vocab_EN.txt'
HI_VOCAB = 'out/vocab_HI.txt'

vocab_en = dict()
vocab_hi = dict()
with open(EN_VOCAB, 'r') as env:
    for line in env.readlines():
        word, nll = line.strip().split()
        vocab_en[word] = float(nll)

with open(HI_VOCAB, 'r') as hiv:
    for line in hiv.readlines():
        word, nll = line.strip().split()
        vocab_hi[word] = float(nll)

sentences = []
with open(sentence_file, 'r') as sentfile:
    for line in sentfile.readlines():
        line = line.strip()
        sentences.append([word for word in line.split(' ')])

outfile = open(output_file, 'w+')
processedfile = open(processed_file, 'w+') # just for visualizing, which word gets which tag
processed_noamb = open(processed_file_noamb, 'w+')

for sentence in sentences:
    for word in sentence:
        if word == '':
            continue
        word = word.lower()
        word_nll_en = vocab_en.get(word, -1)
        word_nll_hi = vocab_hi.get(word, -1)
        mask = None
        if word_nll_en == -1 and word_nll_hi == -1:
            mask = "OTHER"
        elif word_nll_hi == -1 and word_nll_en != -1:
            mask = "EN" # this case seems a bit flimsy, what if its extremely rare in hindi and doesnt occur in english??
        elif word_nll_en == -1 and word_nll_hi != -1:
            mask = "HI"
        elif word_nll_hi + np.log(10) < word_nll_en:
            mask = "HI"
        elif word_nll_en + np.log(10) < word_nll_hi:
            mask = "EN"
        elif np.abs(word_nll_en - word_nll_hi) < np.log(10):
            if word_nll_en < word_nll_hi:
                mask = "AMB-EN"
            else:
                mask = "AMB-HI"
        outfile.write(f'{mask} ')
        # outfile_noamb.write(f'{mask if mask[:3] != "AMB" else mask[-2:]} ')
        processed_noamb.write(word + '\t' + f'{mask if mask[:3] != "AMB" else mask[-2:]}' + '\n')
        processedfile.write(word + '\t' + mask + '\n')
    outfile.write('\n')
    processed_noamb.write('\n')
    processedfile.write('\n')

outfile.close()
processed_noamb.close()
processedfile.close()
