import json
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertTokenizer,
    set_seed,
)

logger = logging.getLogger(__name__)

#add arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='None')
parser.add_argument('--model_name', type=str, default='None')
parser.add_argument('--data_path', type=str, default='None')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


##############################################
# BERTForMaskedLM with Residual Connections  #
##############################################
class ResidualBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, out_embed=768, res_layer=2, res_dropout=0.2) -> None:
        super(ResidualBertForMaskedLM, self).__init__(config)
        self.out_embed = out_embed
        self.res_layer = res_layer
        self.res_dropout = res_dropout
        self.ln = nn.LayerNorm((config.hidden_size,))
        logging.info("ResidualBertForMaskedLM")
        logging.info("RES_LAYER: {}".format(self.res_layer))
        logging.info("DROPOUT: {}".format(self.res_dropout))
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        skip_conn = outputs[2][self.res_layer]
        final_out = outputs[2][-1]
        resid_out = F.dropout(skip_conn, p=self.res_dropout) + final_out
        prediction_scores = self.cls(resid_out)

        outputs = (prediction_scores,) + outputs[2:]

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs


def main():
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    config = BertConfig.from_pretrained(args.model_path, output_hidden_states=True)
    if args.model_name == 'bert':
        model = BertForMaskedLM.from_pretrained(
            args.model_path + '/pytorch_model.bin',
            config=config
        )
    elif args.model_name.startswith('residual-bert'):
        parselist = args.model_name.split('_')
        assert len(parselist) == 3
        res_layer = int(parselist[1])
        res_dropout = float(parselist[2])
        model = ResidualBertForMaskedLM.from_pretrained(
            args.model_path + '/pytorch_model.bin',
            config=config,
            res_layer=res_layer,
            res_dropout=res_dropout
        )
    else:
        raise NotImplementedError
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    # Pseudo Perplexity Evaluation
    ppl_mean, ppl_std = pseudoppl_calc(model, tokenizer, data, device)
    logger.info("PseudoPPL: {:.4f} ± {:.4f}".format(ppl_mean, ppl_std))
    print("PseudoPPL: {:.4f} ± {:.4f}".format(ppl_mean, ppl_std))

    # Cosine Similarity Evaluation
    cos_mean, cos_std = avg_cosine_sim(model, tokenizer, data, device)
    logger.info("Cosine Similarity: {:.4f} ± {:.4f}".format(cos_mean, cos_std))
    print("Cosine Similarity: {:.4f} ± {:.4f}".format(cos_mean, cos_std))


def avg_cosine_sim(model, tokenizer, data, device):
    cosine_means = []
    cosine_stds = []
    for k, v in tqdm(data.items(), total=len(data), desc="Cosine Similarity Calculation"):
        cosine_mean, cosine_std = cosine_sim(model, tokenizer, k, v, device)
        cosine_means.append(cosine_mean)
        cosine_stds.append(cosine_std)
    return np.mean(cosine_means), np.mean(cosine_stds)


def cosine_sim(model, tokenizer, sentence, candidates, device):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    tensor_input = tensor_input.to(device)
    sentence_embedding = model.bert(tensor_input)[0][0][0] # [CLS] token
    cosines = []
    for candidate in candidates:
        candidate_input = tokenizer.encode(candidate, return_tensors='pt').to(device)
        candidate_embedding = model.bert(candidate_input)[0][0][0] # [CLS] token
        cosines.append(F.cosine_similarity(sentence_embedding, candidate_embedding, dim=0).item())
    return np.mean(cosines), np.std(cosines)


def pseudoppl_calc(model, tokenizer, data, device):
    ppl_means = []
    ppl_stds = []
    for _, v in tqdm(data.items(), total=len(data), desc="PseudoPPL Calculation"):
        ppl = []
        for sentence in v:
            ppl.append(pseudoppl(model, tokenizer, sentence, device))
        ppl_means.append(np.mean(ppl))
        ppl_stds.append(np.std(ppl))
    return np.mean(ppl_means), np.mean(ppl_stds)


def pseudoppl(model, tokenizer, sentence, device):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    masked_input, labels = masked_input.to(device), labels.to(device)
    loss = model(masked_input, masked_lm_labels=labels)[0]
    return np.exp(loss.item())


if __name__ == "__main__":
    main()