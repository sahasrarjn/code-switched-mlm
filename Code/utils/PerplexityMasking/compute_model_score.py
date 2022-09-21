"""
Author:         Jey Han Lau
Date:           Jul 19
"""
# outputfile
from joblib import Parallel, delayed
import sys
import argparse
import torch
import math
import pickle
import numpy as np
from tqdm import tqdm
import os
# from calc_corr import get_sentence_data
# from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, XLNetTokenizer, XLNetLMHeadModel, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from scipy.stats.mstats import pearsonr as corr
from scipy.special import softmax

#global
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> """


###########
#functions#
###########
def read_examples_from_file(filename):
    with open(filename, 'r') as infile:
        lines = infile.read().strip().split('\n')
        return [line.strip() for line in lines if line.strip()]


def batched_model_score(tokenize_input, model, tokenizer, device, args):
    tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]
    text_len = len(tokenize_input)
    batch_size = 2000 // text_len
    lp = []
    for batch_id in range(0, text_len, batch_size):
        batched_indexed_tokens = []
        batched_segment_ids = []

        start_idx = batch_id + 1
        end_idx = min(batch_id + 1 + batch_size, text_len + 1)
        for masked_index in range(start_idx, end_idx):
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = '[MASK]'
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)

            segment_ids = [0] * len(tokenize_masked)

            batched_indexed_tokens.append(indexed_tokens)
            batched_segment_ids.append(segment_ids)

        tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
        segment_tensor = torch.tensor(batched_segment_ids, device=device)

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segment_tensor)
            predictions = outputs[0]

        for i, masked_index in enumerate(range(start_idx, end_idx)):
            predicted_score = predictions[i, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            token_idx = tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])
            lp.append(-np.log(predicted_prob[token_idx[0]]))

    return lp


def return_scores(text):
    tokenize_input = []
    subword_splits = []

    for word in text.split(' '):
        toks = tokenizer.tokenize(word)
        tokenize_input += toks
        subword_splits.append(len(toks))

    lp = batched_model_score(tokenize_input, model, tokenizer, device, args)

    pointer = 0
    lp_merged = []
    for length in subword_splits:
        pll = sum(lp[pointer:pointer + length])
        lp_merged.append(str(round(pll, 6)))
        pointer += length

    return lp_merged


def model_score(tokenize_input, model, tokenizer, device, args, tokenize_context=""):

    if args.model_name.startswith("gpt"):

        if not args.use_context:

            #prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
            tensor_input = torch.tensor([[50256] + tokenizer.convert_tokens_to_ids(tokenize_input)], device=device)
            labels = torch.tensor([[50256] + tokenizer.convert_tokens_to_ids(tokenize_input)], device=device)
            labels[:, :1] = -1
            loss = model(tensor_input, labels=tensor_input)

        else:

            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_context + tokenize_input)], device=device)
            labels = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_context + tokenize_input)], device=device)
            #-1 label for context (loss not computed over these tokens)
            labels[:, :len(tokenize_context)] = -1
            loss = model(tensor_input, labels=labels)

        return float(loss[0]) * -1.0 * len(tokenize_input)

    elif args.model_name.startswith("bert") or args.model_name.startswith("xlm"):

        batched_indexed_tokens = []
        batched_segment_ids = []

        if not args.use_context:
            tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]
        else:
            tokenize_combined = ["[CLS]"] + tokenize_context + tokenize_input + ["[SEP]"]
        print(len(tokenize_input))
        for i in range(len(tokenize_input)):

            # Mask a token that we will try to predict back with `BertForMaskedLM`
            masked_index = i + 1 + (len(tokenize_context) if args.use_context else 0)
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = '[MASK]'
            #unidir bert
            #for j in range(masked_index, len(tokenize_combined)-1):
            #    tokenize_masked[j] = '[MASK]'

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segment_ids = [0] * len(tokenize_masked)

            batched_indexed_tokens.append(indexed_tokens)
            batched_segment_ids.append(segment_ids)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
        segment_tensor = torch.tensor(batched_segment_ids, device=device)

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segment_tensor)
            predictions = outputs[0]

        # go through each word and sum their logprobs
        lp = []
        for i in range(len(tokenize_input)):
            masked_index = i + 1 + (len(tokenize_context) if args.use_context else 0)
            predicted_score = predictions[i, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            token_idx = tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])
            lp.append(-np.log(predicted_prob[token_idx[0]]))

        return lp

    elif args.model_name.startswith("xlnet"):

        tokenize_ptext = tokenizer.tokenize(PADDING_TEXT.lower())

        if not args.use_context:
            tokenize_input2 = tokenize_ptext + tokenize_input
        else:
            tokenize_input2 = tokenize_ptext + tokenize_context + tokenize_input

        # go through each word and sum their logprobs
        lp = 0.0
        for max_word_id in range((len(tokenize_input2) - len(tokenize_input)), (len(tokenize_input2))):

            sent = tokenize_input2[:]
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)], device=device)
            perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)

            #if not bidir, mask target word + right/future words
            if not args.xlnet_bidir:
                perm_mask[:, :, max_word_id:] = 1.0
            #if bidir, mask only the target word
            else:
                perm_mask[:, :, max_word_id] = 1.0

            target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
            target_mapping[0, 0, max_word_id] = 1.0

            with torch.no_grad():
                outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
                next_token_logits = outputs[0]

            word_id = tokenizer.convert_tokens_to_ids([tokenize_input2[max_word_id]])[0]
            predicted_prob = softmax((next_token_logits[0][-1]).cpu().numpy())
            lp += np.log(predicted_prob[word_id])

        return lp


######
#main#
######
def main(args):

    #sentence and human ratings
    # sentencexdata = get_sentence_data(args.input_csv)
    # human_ratings = pickle.load(open(args.human_rating_pickle, "rb"))

    #unigram frequencies
    # unigram_freq = pickle.load(open(args.unigram_pickle, "rb"))
    # unigram_total = sum(unigram_freq.values())

    #system scores
    lps = []
    mean_lps = []
    pen_lps = []
    div_lps = []
    sub_lps = []
    slors = []
    pen_slors = []
    sent_ids = []

    #Load pre-trained model and tokenizer
    if args.model_name.startswith("gpt"):
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    elif args.model_name.startswith("bert"):
        if args.first:
            model = BertForMaskedLM.from_pretrained(args.model_name)
        else:
            config = BertConfig.from_pretrained(os.path.join(args.model_loc,'config.json'))
            model = BertForMaskedLM.from_pretrained(os.path.join(args.model_loc,'pytorch_model.bin'), config=config)
        tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=(True if "uncased" in args.model_name else False))
    elif args.model_name.startswith("xlm"):
        print("xlm")
        model = XLMRobertaForMaskedLM.from_pretrained(args.model_name)
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    elif args.model_name.startswith("xlnet"):
        tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
        model = XLNetLMHeadModel.from_pretrained(args.model_name)
    else:
        print("Supported models: gpt, bert and xlnet only.")
        raise SystemExit

    #put model to device (GPU/CPU)
    device = torch.device(args.device)
    model.to(device)

    #eval mode; no dropout
    model.eval()

    #loop through each sentence and compute system scores
    # y = [] #human mean rating
    sent_total = 0
    all_sents = read_examples_from_file(args.input_file)  #["The quick brown fox jumps over the lazy dog"]
    # for sent_id, ratings in tqdm(sorted(human_ratings.items())):

    outfile = open(args.output_file, 'w+')

    max_l = tokenizer.max_len - 2

    for sent in tqdm(all_sents):
        # y.append(np.mean(ratings))

        text = sent  #sentencexdata[sent_id]["SENTENCE"]
        #uppercase first character
        #text = text[0].upper() + text[1:]
        tokenize_input = []
        subword_splits = []
        for word in text.split(' '):
            toks = tokenizer.tokenize(word)
            tokenize_input += toks
            subword_splits.append(len(toks))

        text_len = len(tokenize_input)

        # if args.use_context:
        #     context = sentencexdata[sent_id]["CONTEXT"].replace("\t", " ")
        #     tokenize_context = tokenizer.tokenize(context)
        # else:
        #     tokenize_context = None

        #unigram logprob
        # uni_lp = 0.0
        # for w in tokenize_input:
        #     uni_lp += math.log(float(unigram_freq[w])/unigram_total)

        #compute sentence logprob
        # lp = model_score(tokenize_input, tokenize_context, model, tokenizer, device, args)
        final_lp = []
        for ind in range(0, text_len, max_l):
            lp = batched_model_score(tokenize_input[ind:ind + max_l], model, tokenizer, device, args)
            final_lp.extend(lp)

        pointer = 0
        lp_merged = []
        for length in subword_splits:
            pll = sum(final_lp[pointer:pointer + length])
            lp_merged.append(str(round(pll, 6)))
            pointer += length

        # sent = sent.split()
        # print(len(sent), sent)
        # print(len(lp_merged), lp_merged)
        # print(len(tokenize_input), tokenize_input)
        # print(len(lp), lp)

        #acceptability measures
        # penalty = ((5+text_len)**0.8 / (5+1)**0.8)
        # lps.append(lp)
        # mean_lps.append(lp/text_len)
        # pen_lps.append( lp / penalty )
        # div_lps.append(-lp / uni_lp)
        # sub_lps.append(lp - uni_lp)
        # slors.append((lp - uni_lp) / text_len)
        # pen_slors.append((lp - uni_lp) / penalty)
        # sent_ids.append(sent_id)

        outfile.write(' '.join(lp_merged) + '\n')

        sent_total += 1

    outfile.close()

   

    # results = [corr(lps, y)[0],
    #     corr(mean_lps, y)[0],
    #     corr(pen_lps, y)[0],
    #     corr(div_lps, y)[0],
    #     corr(sub_lps, y)[0],
    #     corr(slors, y)[0],
    #     corr(pen_slors, y)[0]]

    # if args.verbose:
    #print("Correlations:")
    # print("LP     = %.2f" % results[0])
    # print("MeanLP = %.2f" % results[1])
    # print("PenLP  = %.2f" % results[2])
    # print("NormLP = %.2f" % results[3])
    #print("Norm LogProb (Sub) =", results[4])
    # print("SLOR   = %.2f" % results[5])
    #print("SLOR with Length Penalty =", results[6])


if __name__ == "__main__":

    #parser arguments
    desc = "Computes correlation using pytorch transformer models"
    parser = argparse.ArgumentParser(description=desc)

    #arguments
    # parser.add_argument("-r", "--human-rating-pickle", required=True, help="Pickle file containing human ratings")
    # parser.add_argument("-i", "--input-csv", required=True, help="Mturk input csv file containing sentence data")
    parser.add_argument("-m", "--model-name", required=True, help="Pretrained model name (gpt2/gpt2-medium/bert-base-[un]cased/bert-large-[un]cased)")
    parser.add_argument("-l", "--model_loc", help="Pretrained model location")
    parser.add_argument("-s", "--first", action="store_true", help="Pretrained or not")
    # parser.add_argument("-u", "--unigram-pickle", required=True, help="Pickle file containing unigram frequencies (used for SLOR and NormLP)")
    parser.add_argument("-f", "--input_file", required=True, help="File from which to read lines")
    parser.add_argument("-o", "--output_file", required=True, help="File which to write lines")
    parser.add_argument("-c", "--use-context", action="store_true", help="use context at test time for the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose print")
    parser.add_argument("-d", "--device", default="cpu", help="specify the device to use (cpu or cuda:X for gpu); default=cpu")
    parser.add_argument("--xlnet-bidir", action="store_true", help="bidir for xlnet (sees left and right context)")
    args = parser.parse_args()

    main(args)
