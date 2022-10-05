from genericpath import isfile
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


def splitLID(lid):
	'''Return base lid and ambiguity of the token'''
	if lid == "OTHER" or len(lid) == 2:
		return lid, False
	else:
		return lid[-2:], True
		

def isAMB(lid):
	return len(lid) > 3 and int(lid[:3] == "AMB")

def detect_switch_boundary_amb(langs):
	index = [(langs[i], i) for i in range(len(langs)) if langs[i] != 'OTHER']
	marked = np.full(len(langs), -1).tolist()
	for i in range(0, len(index) -1):
		lid1 = splitLID(index[i][0])
		lid2 = splitLID(index[i+1][0])
		if lid1[0] != lid2[0]:
			ambCount = lid1[1] + lid2[1]
			marked[index[i][1]] = max(ambCount, marked[index[i][1]])
			marked[index[i + 1][1]] = max(ambCount, marked[index[i+1][1]])
	return marked

def mask_around_switch_points_amb(words, langs, tokenMap):
	markedLangs = detect_switch_boundary_amb(langs)
	markedLangs = {words[i]:markedLangs[i] for i in range(len(langs))}
	maskTokens = []
	for word in words:
		if markedLangs[word] == -1:
			mask = "NOMASK"
		else:
			mask = "MASK" + str(markedLangs[word])
		fill = np.full(len(tokenMap[word]), mask).tolist()
		maskTokens.extend(fill)
	return maskTokens

if args.mask_type == "around-switch-amb":
	implement_mask = mask_around_switch_points_amb
else: 
	raise ValueError("Unsupported masking style, please re-check your arguments")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

if not args.debug: g =  open(args.target, 'w+') 

engSum = 0
count = 0
numSwitch = 0
maskRatio = [0, 0, 0]

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
	# for l in lines:
		if l == "":
		#	engSum = engSum + tokenLangs.count('EN')/len(tokenLangs)
		#	count = count + 1
			check = list(set(tokenLangs))
			if 'OTHER' in check: 
				check.remove('OTHER')
			if  len(check) > 1: 
				engSum = engSum + tokenLangs.count('EN')/len(tokenLangs)
				count = count + 1
				numSwitch = numSwitch + (len([x[0] for x in groupby(tokenLangs)]) -1)/len(words)
				if args.mask_type == "around-switch-amb":
					tokenMasks = implement_mask(words, langs, tokenMap)
				else:
					tokenMasks = implement_mask(tokenLangs)
				assert len(tokenMasks) == len(tokens), "Mask length doesnot match length of tokens"
				sentence = " ".join(words)
				sentence_tokenized = tokenizer.tokenize(sentence)
				assert len(tokens) == len(sentence_tokenized), "Mismatch in lengths of sentence and combined word-level tokens"
				#if tokens != sentence_tokenized:
					#print("{}, {}".format(tokens, sentence_tokenized))
					#pdb.set_trace()
				labelSent = ' '.join(tokenMasks)
				
				for i in range(3):
					maskRatio[i] = maskRatio[i] + tokenMasks.count('MASK'+str(i))/len(tokenMasks)
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
			print(l)
			print(words, langs)
			raise Exception('Invalid sep token')
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


print("For {} sentences, average fraction of english tokens is {}, and average number of switch points are {}. The average mask ratio is MASK0:{}, MASK1:{}, MASK2:{}, MASK#:{}".format(
	count, 
	np.round(engSum/count, 2), 
	np.round(numSwitch/count,2), 
	np.round(maskRatio[0]/count,2), 
	np.round(maskRatio[1]/count,2), 
	np.round(maskRatio[2]/count,2), 
	np.round(sum(maskRatio)/count,2))
)
