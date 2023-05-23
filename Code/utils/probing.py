import os
import json
import wandb
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)

from customlibs.custom_language_modeling_probing import LineByLineProbeDataset

from customlibs.residual_bert import ResidualBertModel

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name: Optional[str] = field(
        default="bert",
        metadata={"help": "Choose from bert or residual-bert_<layer>_<dropout>"},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the pretrained weights for the model. This is the path to the baseline if 'conditional-probing'"
        },
    )
    model_conditional_name: Optional[str] = field(
        default="bert",
        metadata={"help": "Choose from bert or residual-bert_<layer>_<dropout>"},
    )
    model_conditional_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the pretrained weights for the model. Path to the specialized model like switchMLM/freqMLM"
        },
    )
    probe_layer: int = field(
        default=-1, metadata={"help": "Choose layer to probe, default -> last layer"}
    )
    model_type: Optional[str] = field(
        default="linear-head",
        metadata={"help": "Choose from linear-head or bilstm-head or mlp-head or conditional-linear-head or conditional-mlp-head"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    wandb: bool = field(
        default=False,
        metadata={"help": "specify to use wandb logging"},
    )
    experiment_name: Optional[str] = field(
        default="en_hi_probing_experiment",
        metadata={"help": "display name of the experiment on wandb"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file).", "nargs": "+"}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file).", "nargs": "+"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    files = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineProbeDataset(tokenizer=tokenizer, files=files, block_size=args.block_size)


class BertBiLSTMHead(nn.Module):
    def __init__(self, bert, out_embed, hidden_dim, probe_layer) -> None:
        super(BertBiLSTMHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.probe_layer = probe_layer
        self.bert = bert
        self.lstm = nn.LSTM(out_embed, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, 2)
    
    def forward(self, inp):
        bert_embedding = self.bert(inp)[2][self.probe_layer]
        lstm_out, (_, _) = self.lstm(bert_embedding)
        linear_output = self.linear(lstm_out)
        logprobs = F.log_softmax(linear_output, dim=-1)
        return logprobs


class BertLinearHead(nn.Module):
    def __init__(self, bert, out_embed, probe_layer) -> None:
        super(BertLinearHead, self).__init__()
        self.probe_layer = probe_layer
        self.bert = bert
        self.linear = nn.Linear(out_embed, 2)
    
    def forward(self, inp):
        bert_embedding = self.bert(inp)[2][self.probe_layer]
        linear_output = self.linear(bert_embedding)
        logprobs = F.log_softmax(linear_output, dim=-1)
        return logprobs


class BertConditionalLinearHead(nn.Module):
    def __init__(self, bert1, bert2=None, out_embed=768) -> None:
        super(BertConditionalLinearHead, self).__init__()
        self.bert1 = bert1
        self.bert2 = bert2
        self.linear = nn.Linear(2*out_embed, 2)
    
    def forward(self, inp):
        bert1_embedding = self.bert1(inp)[0]
        if self.bert2 is not None:
            bert2_embedding = self.bert2(inp)[0]
        else:
            bert2_embedding = torch.zeros_like(bert1_embedding)
        concat_embed = torch.cat((bert1_embedding, bert2_embedding), dim=-1)
        linear_output = self.linear(concat_embed)
        logprobs = F.log_softmax(linear_output, dim=-1)
        return logprobs


class BertMLPHead(nn.Module):
    def __init__(self, bert, out_embed, int_embed, probe_layer) -> None:
        super(BertMLPHead, self).__init__()
        self.probe_layer = probe_layer
        self.bert = bert
        self.linear1 = nn.Linear(out_embed, int_embed)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int_embed, 2)
    
    def forward(self, inp):
        bert_embedding = self.bert(inp)[2][self.probe_layer]
        linear_output = self.linear1(bert_embedding)
        nonlin_output = self.relu(linear_output)
        linear_output = self.linear2(nonlin_output)
        logprobs = F.log_softmax(linear_output, dim=-1)
        return logprobs


class BertConditionalMLPHead(nn.Module):
    def __init__(self, bert1, bert2=None, out_embed=768, int_embed=512) -> None:
        super(BertConditionalMLPHead, self).__init__()
        self.bert1 = bert1
        self.bert2 = bert2
        self.linear1 = nn.Linear(2*out_embed, int_embed)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int_embed, 2)
    
    def forward(self, inp):
        bert1_embedding = self.bert1(inp)[0]
        if self.bert2 is not None:
            bert2_embedding = self.bert2(inp)[0]
        else:
            bert2_embedding = torch.zeros_like(bert1_embedding)
        concat_embed = torch.cat((bert1_embedding, bert2_embedding), dim=-1)
        linear_output1 = self.linear1(concat_embed)
        nonlin_output = self.relu(linear_output1)
        linear_output2 = self.linear2(nonlin_output)
        logprobs = F.log_softmax(linear_output2, dim=-1)
        return logprobs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_conditional_path == 'None':
        model_args.model_conditional_path = None

    # WANDB STUFF
    if model_args.wandb:
        wandb.init(project="MLMPretraining-Probing", entity="csalt-pretraining", name=model_args.experiment_name)

    if not data_args.eval_data_file and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    config1 = BertConfig.from_pretrained(model_args.model_path, output_hidden_states=True)

    if model_args.model_name == 'bert':
        bert1 = BertModel.from_pretrained(
            model_args.model_path + '/pytorch_model.bin',
            config=config1
        )
    elif model_args.model_name.startswith('residual-bert'):
        parselist = model_args.model_name.split('_')
        assert len(parselist) == 3
        res_layer = int(parselist[1])
        res_dropout = float(parselist[2])
        bert1 = ResidualBertModel.from_pretrained(
            model_args.model_path + '/pytorch_model.bin',
            config=config1,
            res_layer=res_layer,
            res_dropout=res_dropout
        )
    bert1.resize_token_embeddings(len(tokenizer))

    # Freeze bert layers
    for name, param in bert1.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    # Load specialized model if present for conditional probing
    bert2 = None
    if model_args.model_conditional_path is not None:
        config2 = BertConfig.from_pretrained(model_args.model_conditional_path)
        if model_args.model_conditional_name == 'bert':
            bert2 = BertModel.from_pretrained(
                model_args.model_conditional_path + '/pytorch_model.bin',
                config=config2
            )
        elif model_args.model_conditional_name.startswith('residual-bert'):
            parselist = model_args.model_conditional_name.split('_')
            assert len(parselist) == 3
            res_layer = int(parselist[1])
            res_dropout = float(parselist[2])
            bert2 = ResidualBertModel.from_pretrained(
                model_args.model_conditional_path + '/pytorch_model.bin',
                config=config2,
                res_layer=res_layer,
                res_dropout=res_dropout
            )
        bert2.resize_token_embeddings(len(tokenizer))
        # Freeze bert layers
        for name, param in bert2.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    if model_args.model_type == "bilstm-head":
        model = BertBiLSTMHead(bert1, 768, 256, model_args.probe_layer)
    elif model_args.model_type == "linear-head":
        model = BertLinearHead(bert1, 768, model_args.probe_layer)
    elif model_args.model_type == "mlp-head":
        model = BertMLPHead(bert1, 768, 256, model_args.probe_layer)
    elif model_args.model_type == "conditional-linear-head":
        model = BertConditionalLinearHead(bert1, bert2, 768)
    elif model_args.model_type == "conditional-mlp-head":
        model = BertConditionalMLPHead(bert1, bert2, 768, 512)
    model.to(device)

    if data_args.block_size <= 0:
        # Our input block size will be the max possible for the model
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    EPOCHS = int(training_args.num_train_epochs)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    clip = 5
    
    EPOCH_METRICS = {
        "val_losses": [],
        "val_acc_hard_hamming": [],
        "val_acc_soft_hamming": []
    }

    # Eval at start
    logger.info(f"EVAL AT START")
    val_losses, hhtable, shtable = evaluate(model, eval_dataset, loss_fn, device)

    logger.info(f"Val Loss: {np.mean(val_losses)}")
    logger.info(f"Val Hard Hamming Accuracy: {np.sum(hhtable[:,0])/np.sum(hhtable[:,1])}")
    logger.info(f"Val Soft Hamming Accuracy: {np.sum(shtable[:,0])/np.sum(shtable[:,1])}")
    if model_args.wandb:
        wandb.log({"val_loss": np.mean(val_losses)})
        wandb.log({"val_acc_hard_hamming": np.sum(hhtable[:,0])/np.sum(hhtable[:,1])})
        wandb.log({"val_acc_soft_hamming": np.sum(shtable[:,0])/np.sum(shtable[:,1])})
        wandb.log({"steps": 0})

    steps = 0
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        for sentence, probe_target in tqdm(train_dataset, desc="Train Samples"):
            if steps >= training_args.max_steps:
                break
            sentence, probe_target = sentence.to(device), probe_target.to(device)
            model.zero_grad()
            model_pred = model(sentence.unsqueeze(0)).squeeze(0)
            loss = loss_fn(model_pred, probe_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            steps += 1
            if steps%training_args.logging_steps == 0:
                # Eval after logging steps
                logger.info(f"EVAL AFTER STEPS {steps} EPOCH {epoch+1}")
                val_losses, hhtable, shtable = evaluate(model, eval_dataset, loss_fn, device)

                EPOCH_METRICS["val_losses"].append(np.mean(val_losses))
                EPOCH_METRICS["val_acc_hard_hamming"].append(np.sum(hhtable[:,0])/np.sum(hhtable[:,1]))
                EPOCH_METRICS["val_acc_soft_hamming"].append(np.sum(shtable[:,0])/np.sum(shtable[:,1]))
                logger.info(f"Steps: {steps}\n")
                logger.info(f"Epoch: {epoch+1}/{EPOCHS}\n")
                logger.info(f"Val Loss: {EPOCH_METRICS['val_losses'][-1]}")
                logger.info(f"Val Hard Hamming Accuracy: {EPOCH_METRICS['val_acc_hard_hamming'][-1]}")
                logger.info(f"Val Soft Hamming Accuracy: {EPOCH_METRICS['val_acc_soft_hamming'][-1]}")
                if model_args.wandb:
                    wandb.log({"val_loss": EPOCH_METRICS['val_losses'][-1]})
                    wandb.log({"val_acc_hard_hamming": EPOCH_METRICS["val_acc_hard_hamming"][-1]})
                    wandb.log({"val_acc_soft_hamming": EPOCH_METRICS["val_acc_soft_hamming"][-1]})
                    wandb.log({"steps": steps})                

    # Eval at end
    logger.info(f"EVAL AT END")
    val_losses, hhtable, shtable = evaluate(model, eval_dataset, loss_fn, device)
    
    EPOCH_METRICS["val_losses"].append(np.mean(val_losses))
    EPOCH_METRICS["val_acc_hard_hamming"].append(np.sum(hhtable[:,0])/np.sum(hhtable[:,1]))
    EPOCH_METRICS["val_acc_soft_hamming"].append(np.sum(shtable[:,0])/np.sum(shtable[:,1]))
    logger.info(f"Steps: {steps}\n")
    logger.info(f"Val Loss: {EPOCH_METRICS['val_losses'][-1]}")
    logger.info(f"Val Hard Hamming Accuracy: {EPOCH_METRICS['val_acc_hard_hamming'][-1]}")
    logger.info(f"Val Soft Hamming Accuracy: {EPOCH_METRICS['val_acc_soft_hamming'][-1]}")
    if model_args.wandb:
        wandb.log({"val_loss": EPOCH_METRICS['val_losses'][-1]})
        wandb.log({"val_acc_hard_hamming": EPOCH_METRICS['val_acc_hard_hamming'][-1]})
        wandb.log({"val_acc_soft_hamming": EPOCH_METRICS['val_acc_soft_hamming'][-1]})
        wandb.log({"steps": steps})

    logger.info(f"SAVING FINAL MODEL TO: {training_args.output_dir + '/state_dict.pt'}")
    torch.save(model.state_dict(), training_args.output_dir + '/state_dict.pt')
    logger.info(f"EPOCH METRICS: val_losses: {EPOCH_METRICS['val_losses']}")
    logger.info(f"EPOCH METRICS: val_acc_hard_hamming: {EPOCH_METRICS['val_acc_hard_hamming']}")
    logger.info(f"EPOCH METRICS: val_acc_soft_hamming: {EPOCH_METRICS['val_acc_soft_hamming']}")
    print("EPOCH METRICS: ", EPOCH_METRICS)
    with open(training_args.output_dir + '/metrics.json', 'w+') as mfp:
        mfp.write(json.dumps(EPOCH_METRICS))


def evaluate(model, eval_dataset, loss_fn, device):
    model.eval()
    val_losses = []
    # number of predicted correctly, total tokens in each sentence : Hard Hamming Distance
    hard_hamming_table = np.zeros((len(eval_dataset), 2))
    # number of probabilistically correct, total tokens in each sentence: Soft Hamming Distance
    soft_hamming_table = np.zeros((len(eval_dataset), 2)) 
    for idx, (sent, target) in tqdm(enumerate(eval_dataset), desc="Evaluation Samples"):
        sent, target = sent.to(device), target.to(device)
        model_pred = model(sent.unsqueeze(0)).squeeze(0)
        # These are the log probabilities returned by the model

        val_losses += [loss_fn(model_pred, target).item()]
        hard_hamming_table[idx][0] = torch.sum(torch.argmax(model_pred, dim=-1)==target)
        hard_hamming_table[idx][1] = target.shape[-1]

        soft_hamming_table[idx][0] = torch.sum(torch.exp(model_pred[torch.arange(target.shape[-1]), target]))
        soft_hamming_table[idx][1] = target.shape[-1]
    model.train()
    return val_losses, hard_hamming_table, soft_hamming_table


if __name__ == "__main__":
    main()
