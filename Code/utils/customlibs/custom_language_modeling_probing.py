import os
import pdb
import torch
import logging

from torch.utils.data.dataset import Dataset
from customlibs.custom_tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class LineByLineProbeDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, files: list, block_size: int):
		self.examples = [] 
		self.target = []
		for file_path in files:
			print(file_path)
			assert os.path.isfile(file_path)

			logger.info("Creating features from dataset file at %s", file_path)
			with open(file_path, encoding="utf-8") as f:
				lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

			target = ['0 {} 0'.format(' '.join(line.split('\t')[-1].split()[:block_size - 2])).split() for line in lines] # Pre-computed token level mask indicators, deal with [CLS] and [SEP] tokens
			lines = [line.split('\t')[0] for line in lines]
			target = [[int(msk) for msk in t] for t in target]
					
			batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
			examples = batch_encoding["input_ids"]
			
			for j in range(len(examples)):
				if len(examples[j]) != len(target[j]):
					print(examples[j], lines[j])
					print(f'{j}***{len(examples[j])}***{len(target[j])}')
					break

			for j in range(len(examples)): 
				assert len(examples[j]) == len(target[j]),  pdb.set_trace() #"Tokenized sentence and token-mask indicators are of different lengths, {} and {} for {}".format(len(self.examples[j]), len(target[j]), lines[j])
			
			self.examples.extend(examples)
			self.target.extend(target)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.target[i], dtype=torch.long)


class LineByLineTextProbeDataset(Dataset):
	"""
		Handling both probe and MLM data in the same Dataset
	"""
	def __init__(self, tokenizer: PreTrainedTokenizer, files: list, block_size: int):
		self.examples = [] 
		self.probetarget = []
		self.tomasks = []
		self.classes = []
		for i, file_path in enumerate(files):
			print(file_path)
			assert os.path.isfile(file_path)

			logger.info("Creating features from dataset file at %s", file_path)
			with open(file_path, encoding="utf-8") as f:
				lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

			tomasks = ['NOMASK {} NOMASK'.format(' '.join(line.split('\t')[-1].split()[:block_size - 2])).split() for line in lines] # Pre-computed token level mask indicators, deal with [CLS] and [SEP] tokens
			lines = [line.split('\t')[0] for line in lines]
			tomasks = [[msk == "MASK" for msk in t] for t in tomasks]
			probetarget = [[int(msk) for msk in t] for t in tomasks]

			batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
			examples = batch_encoding["input_ids"]
			
			for j in range(len(examples)):
				if len(examples[j]) != len(probetarget[j]):
					print(examples[j], lines[j])
					print(f'{j}***{len(examples[j])}***{len(probetarget[j])}')
					break

			for j in range(len(examples)): 
				assert len(examples[j]) == len(probetarget[j]),  pdb.set_trace() #"Tokenized sentence and token-mask indicators are of different lengths, {} and {} for {}".format(len(self.examples[j]), len(target[j]), lines[j])
			
			self.examples.extend(examples)
			self.probetarget.extend(probetarget)
			self.tomasks.extend(tomasks)
			self.classes.extend([i] * len(examples))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return (torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.probetarget[i], dtype=torch.long), torch.tensor(self.tomasks[i], dtype=torch.bool), self.classes[i])
