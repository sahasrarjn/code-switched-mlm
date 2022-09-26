import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import groupby
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, help="Dir of source file(s)/File")
parser.add_argument("--dev", default=False, action="store_true", help="To only consider dev files")
parser.add_argument("--all", default=False, action="store_true", help="To consider all files")
parser.add_argument("--debug", default=False, action="store_true", help="Launches the debugger")
parser.add_argument("--target", type=str, help="Name of the dest file")
parser.add_argument("--tokenizer-name", type=str, help="Name of tokenizer")
parser.add_argument("--mask-type", type=str, help="Type of masking to be implemented")
args = parser.parse_args()

def mask_english_tokens(langTokens):
	maskTokens = ["MASK" if l == "EN" else "NOMASK" for l in langTokens]
	return maskTokens

def mask_all_tokens(langTokens):
	maskTokens = ["MASK" for l in langTokens]
	return maskTokens

def mask_english_hindi_tokens(langTokens):
	maskTokens =  ["NOMASK" if l == "OTHER"  else "MASK" for l in langTokens]
	return maskTokens

def detect_switch_boundary(langs):
	index = [(langs[i], i) for i in range(len(langs)) if langs[i] != 'OTHER']
	marked = np.full(len(langs), False).tolist()
	for i in range(0, len(index) -1):
		if index[i][0] != index[i + 1][0]:
			marked[index[i][1]] = True
			marked[index[i + 1][1]] = True
	return marked

def mask_around_switch_points(words, langs, tokenMap):
	markedLangs = detect_switch_boundary(langs)
	markedLangs = {words[i]:markedLangs[i] for i in range(len(langs))}
	maskTokens = []
	for word, lang in zip(words, langs):
		if lang == 'OTHER':
			fill = np.full(len(tokenMap[word]), "MASK").tolist()
		else:
			fill = np.full(len(tokenMap[word]), "MASK").tolist() if markedLangs[word] else np.full(len(tokenMap[word]), "NOMASK").tolist()
		maskTokens.extend(fill)
	return maskTokens
	
if args.mask_type == "english-only":
	implement_mask = mask_english_tokens
elif args.mask_type == "all-tokens":
	implement_mask = mask_all_tokens
elif args.mask_type == "english-hindi":
	implement_mask = mask_english_hindi_tokens
elif args.mask_type == "around-switch":
	implement_mask = mask_around_switch_points

else : raise ValueError("Unsupported masking style, please re-check your arguments")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

if not args.debug: g =  open(args.target, 'w+') 

engSum = 0
count = 0
numSwitch = 0
maskRatio = 0

def processDataset(source_file, engSum, count, numSwitch, maskRatio):
	f = open(source_file, 'r')
	lines = f.readlines()
	f.close()

	lines = [l.strip() for l in lines]

	words = []
	langs = []
	tokens = []
	tokenLangs = []
	tokenMap = {}

	for l in tqdm(lines):

		if l == "":
			check = list(set(tokenLangs))
			# print(words, langs)
			# print(tokens, tokenLangs)
			# print(check)
			if 'OTHER' in check: 
				check.remove('OTHER')
			if  len(check) > 1: 
				engSum = engSum + tokenLangs.count('EN')/len(tokenLangs)
				count = count + 1
				numSwitch = numSwitch + (len([x[0] for x in groupby(tokenLangs)]) -1)/len(words)
				if args.mask_type == "around-switch":
					tokenMasks = implement_mask(words, langs, tokenMap)
				else:
					tokenMasks = implement_mask(tokenLangs)
				assert len(tokenMasks) == len(tokens), "Mask length doesnot match length of tokens"
				sentence = " ".join(words)
				sentence_tokenized = tokenizer.tokenize(sentence)
				assert len(tokens) == len(sentence_tokenized), "Mismatch in lengths of sentence and combined word-level tokens"
				
				labelSent = ' '.join(tokenMasks)
				maskRatio = maskRatio + tokenMasks.count('MASK')/len(tokenMasks)
				if args.debug:
					pass
				else:
					g.write('{}\t{}\n'.format(sentence, labelSent))
			words, langs, tokens, tokenLangs = [],[],[],[]
			tokenMap = {}
			continue
		try:
			word, label = l.split('\t')
		except:
			print(words, langs)
			raise Exception('Unexpected input line with space separated word and label: {}', l)
		if word  == '' or word == " ": continue
		words.append(word)
		langs.append(label)
		tokenized = tokenizer.tokenize(word)
		tokens.extend(tokenized)
		tokenLabels = np.full(len(tokenized),label).tolist()
		tokenLangs.extend(tokenLabels)
		tokenMap[word] = tokenized

	return engSum, count, numSwitch, maskRatio

if os.path.isfile(args.source):
	files = [args.source]
else:
	files = os.listdir(args.source)
	train_files = [os.path.join(args.source, f) for f in files if 'dev' not in f]
	dev_files = [os.path.join(args.source, f) for f in files if 'dev' in f]

if args.dev:
	files = dev_files
elif not args.all and not args.dev and not os.path.isfile(args.source):
	files = train_files

if args.all:
	files = [os.path.join(args.source, f) for f in files]

for file in files:
	print(file)
	if file[-4:] != '.txt': continue
	engSum, count, numSwitch, maskRatio = processDataset(file, engSum, count, numSwitch, maskRatio)

if not args.debug: g.close()


print("For {} sentences, average fraction of english tokens is {}, and average number of switch points are {}. The average mask ratio is {}".format(count, np.round(engSum/count, 2), np.round(numSwitch/count,2), np.round(maskRatio/count,2)))
