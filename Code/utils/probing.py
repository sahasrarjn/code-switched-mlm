import os
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

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default="linear-head",
        metadata={"help": "Choose from linear-head or bilstm-head"},
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
    def __init__(self, bert, out_embed, hidden_dim) -> None:
        super(BertBiLSTMHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert = bert
        self.lstm = nn.LSTM(out_embed, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, 2)
    
    def forward(self, inp):
        bert_embedding, _ = self.bert(inp)
        lstm_out, (_, _) = self.lstm(bert_embedding)
        linear_output = self.linear(lstm_out)
        probs = F.log_softmax(linear_output, dim=-1)
        return probs


class BertLinearHead(nn.Module):
    def __init__(self, bert, out_embed) -> None:
        super(BertLinearHead, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(out_embed, 2)
    
    def forward(self, inp):
        bert_embedding, _ = self.bert(inp)
        linear_output = self.linear(bert_embedding)
        probs = F.log_softmax(linear_output, dim=-1)
        return probs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    config = BertConfig.from_pretrained(model_args.model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert = BertModel.from_pretrained(
        model_args.model_path + '/pytorch_model.bin',
        config=config
    )

    bert.resize_token_embeddings(len(tokenizer))

    # Freeze bert layers
    for name, param in bert.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    if model_args.model_type == "bilstm-head":
        model = BertBiLSTMHead(bert, 768, 256)
    elif model_args.model_type == "linear-head":
        model = BertLinearHead(bert, 768)
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
        "val_accs": []
    }

    # Eval at start
    logger.info(f"EVAL AT START")
    val_losses, accuracy_table = evaluate(model, eval_dataset, loss_fn, device)

    logger.info(f"Val Loss: {np.mean(val_losses)}")
    logger.info(f"Val Accuracy: {np.sum(accuracy_table[:,0])/np.sum(accuracy_table[:,1])}")
    if model_args.wandb:
        wandb.log({"val_loss": np.mean(val_losses)})
        wandb.log({"val_acc": np.sum(accuracy_table[:,0])/np.sum(accuracy_table[:,1])})
        wandb.log({"steps": 0})

    steps = 0
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        for sentence, probe_target in tqdm(train_dataset, desc="Train Samples"):
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
                val_losses, accuracy_table = evaluate(model, eval_dataset, loss_fn, device)

                EPOCH_METRICS["val_losses"].append(np.mean(val_losses))
                EPOCH_METRICS["val_accs"].append(np.sum(accuracy_table[:,0])/np.sum(accuracy_table[:,1]))
                logger.info(f"Steps: {steps}\n")
                logger.info(f"Epoch: {epoch+1}/{EPOCHS}\n")
                logger.info(f"Val Loss: {EPOCH_METRICS['val_losses'][-1]}")
                logger.info(f"Val Accuracy: {EPOCH_METRICS['val_accs'][-1]}")
                if model_args.wandb:
                    wandb.log({"val_loss": EPOCH_METRICS['val_losses'][-1]})
                    wandb.log({"val_acc": EPOCH_METRICS['val_accs'][-1]})
                    wandb.log({"steps": steps})

    # Eval at end
    logger.info(f"EVAL AFTER STEPS {steps} EPOCH {epoch+1}")
    val_losses, accuracy_table = evaluate(model, eval_dataset, loss_fn, device)
    
    EPOCH_METRICS["val_losses"].append(np.mean(val_losses))
    EPOCH_METRICS["val_accs"].append(np.sum(accuracy_table[:,0])/np.sum(accuracy_table[:,1]))
    logger.info(f"Steps: {steps}\n")
    logger.info(f"Epoch: {epoch+1}/{EPOCHS}\n")
    logger.info(f"Val Loss: {EPOCH_METRICS['val_losses'][-1]}")
    logger.info(f"Val Accuracy: {EPOCH_METRICS['val_accs'][-1]}")
    if model_args.wandb:
        wandb.log({"val_loss": EPOCH_METRICS['val_losses'][-1]})
        wandb.log({"val_acc": EPOCH_METRICS['val_accs'][-1]})
        wandb.log({"steps": steps})

    logger.info(f"SAVING FINAL MODEL TO: {training_args.output_dir + '/state_dict.pt'}")
    torch.save(model.state_dict(), training_args.output_dir + '/state_dict.pt')
    logger.info(f"EPOCH METRICS: val_losses: {EPOCH_METRICS['val_losses']}")
    logger.info(f"EPOCH METRICS: val_accs: {EPOCH_METRICS['val_accs']}")
    print("EPOCH METRICS: ", EPOCH_METRICS)


def evaluate(model, eval_dataset, loss_fn, device):
    model.eval()
    val_losses = []
    accuracy_table = np.zeros((len(eval_dataset), 2)) # number of predicted correctly, total tokens in each sentence
    for idx, (sent, target) in tqdm(enumerate(eval_dataset), desc="Evaluation Samples"):
        sent, target = sent.to(device), target.to(device)
        model_pred = model(sent.unsqueeze(0)).squeeze(0)

        val_losses += [loss_fn(model_pred, target).item()]
        accuracy_table[idx][0] = torch.sum(torch.argmax(model_pred, dim=-1)==target)
        accuracy_table[idx][1] = target.shape[-1]
    model.train()
    return val_losses, accuracy_table


if __name__ == "__main__":
    main()
