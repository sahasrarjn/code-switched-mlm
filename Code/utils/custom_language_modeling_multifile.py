import logging
import os
import pickle
import time
import re
import pdb

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from custom_tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
	"""
	This will be superseded by a framework-agnostic approach
	soon.
	"""

	def __init__(
		self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
	):
		assert os.path.isfile(file_path)

		block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

		directory, filename = os.path.split(file_path)
		cached_features_file = os.path.join(
			directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
		)

		# Make sure only the first process in distributed training processes the dataset,
		# and the others will use the cache.
		lock_path = cached_features_file + ".lock"
		with FileLock(lock_path):

			if os.path.exists(cached_features_file) and not overwrite_cache:
				start = time.time()
				with open(cached_features_file, "rb") as handle:
					self.examples = pickle.load(handle)
				logger.info(
					f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
				)

			else:
				logger.info(f"Creating features from dataset file at {directory}")

				self.examples = []
				with open(file_path, encoding="utf-8") as f:
					text = f.read()

				tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
				print(tokenized_text)

				for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
					self.examples.append(
						tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
					)
				# Note that we are losing the last truncated example here for the sake of simplicity (no padding)
				# If your dataset is small, first you should loook for a bigger one :-) and second you
				# can change this behavior by adding (model specific) padding.

				start = time.time()
				with open(cached_features_file, "wb") as handle:
					pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
				logger.info(
					"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
				)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
	"""
	This will be superseded by a framework-agnostic approach
	soon.
	"""

	def __init__(self, tokenizer: PreTrainedTokenizer, files: list, block_size: int):
		self.examples = [] 
		self.tomasks = []
		self.classes = []
		for i, file_path in enumerate(files):
			print(file_path)
			assert os.path.isfile(file_path)

			# Here, we do not cache the features, operating under the assumption
			# that we will soon use fast multithreaded tokenizers from the
			# `tokenizers` repo everywhere =)
			
			logger.info("Creating features from dataset file at %s", file_path)
			with open(file_path, encoding="utf-8") as f:
				lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

			tomasks = ['NOMASK {} NOMASK'.format(' '.join(line.split('\t')[-1].split()[:block_size - 2])).split() for line in lines] # Pre-computed token level mask indicators, deal with [CLS] and [SEP] tokens
			lines = [line.split('\t')[0] for line in lines]
			tomasks = [[msk == "MASK" for msk in t] for t in tomasks]
					
			batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
			examples = batch_encoding["input_ids"]
			
			for j in range(len(examples)):
				if len(examples[j]) != len(tomasks[j]):
					print(examples[j], lines[j])
					print(f'{j}***{len(examples[j])}***{len(tomasks[j])}')
					break

			for j in range(len(examples)): 
				assert len(examples[j]) == len(tomasks[j]),  pdb.set_trace() #"Tokenized sentence and token-mask indicators are of different lengths, {} and {} for {}".format(len(self.examples[j]), len(tomasks[j]), lines[j])
			
			self.examples.extend(examples)
			self.tomasks.extend(tomasks)
			self.classes.extend([i] * len(examples))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.tomasks[i], dtype=torch.bool), self.classes[i]
