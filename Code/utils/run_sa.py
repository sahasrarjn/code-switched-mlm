# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import wandb
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    XLMConfig,
    XLMTokenizer,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    CONFIG_MAPPING
)
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

from customlibs.residual_bert import ResidualBertForSequenceClassification

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(
        y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    return{
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall
    }


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, 'r', errors='ignore') as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        text = x[0]
        if mode!='test':
            label = x[1]
        else:
            label='positive'
        examples.append({'text': text, 'label': label})
    if mode == 'test':
        for i in range(len(examples)):
            if examples[i]['text'] == 'not found':
                examples[i]['present'] = False
            else:
                examples[i]['present'] = True
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 max_seq_length=128):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):

        sentence = example['text']
        label = example['label']

        sentence_tokens = tokenizer.tokenize(sentence)[:max_seq_length - 2]
        sentence_tokens = [tokenizer.cls_token] + \
            sentence_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        label = label_map[label]
        features.append({'input_ids': input_ids,
                         'label': label})
        if 'present' in example:
            features[-1]['present'] = example['present']

    return features


def get_labels(data_dir):
    # return ['0','1','2']
    # return ['0.0','1.0','2.0']
    # return ['negative','positive','neutral']
    all_path = os.path.join(data_dir, "all.txt")
    labels = []
    with open(all_path, "r", errors='ignore') as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        splits = line.split('\t')
        label = splits[-1]
        if label not in labels:
            labels.append(label)
    return labels


def train(args, train_dataset, valid_dataset, model, tokenizer, labels):

    # Prepare train data
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate)
    train_batch_size = args.train_batch_size

    # Prepare optimizer
    t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(t_total // 10 if True else 0), num_training_steps=t_total) #set to match NLI gluecose repo

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_f1_score = 0
    epnum=0
    all_tests=[]
    val_losses = []
    train_losses = []
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        t_loss = 0
        t_steps = 0
        for step, batch in enumerate(epoch_iterator):
            # print("Step: ", step)
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            t_loss+=loss.item()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            t_steps+=1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # print("Global step:", global_step, step)
                if (global_step%args.logging_steps==0) :
                    results, _ = evaluate(args, model, tokenizer, labels, 'validation')
                    if results.get('f1') > best_f1_score and args.save_steps > 0:
                        best_f1_score = results.get('f1')
                        logger.info(" new Best F1 = %s", str(best_f1_score))
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(
                            args.output_dir, "training_args.bin"))
                        print(f"Model saved to {args.output_dir}")
                        try:
                            preds = evaluate(args, model, tokenizer, labels, mode="test")
                            all_tests.append((results.get('f1'),preds))
                            if args.wandb: 
                                wandb.log({'all_tests_f1': results.get('f1')})
                        except Exception as e:
                            print(e)
                    elif results.get('f1') > best_f1_score and args.save_steps == 0:
                        best_f1_score=results.get('f1')
                        logger.info(" new Best F1 = %s", str(best_f1_score))
                        try:
                            preds = evaluate(args, model, tokenizer, labels, mode="test")
                            all_tests.append((results.get('f1'),preds))
                            if args.wandb: 
                                wandb.log({'all_tests_f1': results.get('f1')})
                        except Exception as e:
                            print(e)
                    else:
                        logger.info("  Best F1 still at = %s", str(best_f1_score))

                    if args.wandb:
                        wandb.log({'f1': results.get('f1')*100})
                        wandb.log({'best_f1': best_f1_score*100})
                        
                    # if results.get('f1') > best_f1_score:
                    #     best_f1_score = results.get('f1')
                    #     preds = evaluate(args, model, tokenizer, labels, mode="test")
                    #     # Saving predictions
                    #     output_test_predictions_file = os.path.join(args.output_dir, "test_predictions_"+str(best_f1_score)+"_.txt")
                    #     with open(output_test_predictions_file, "w") as writer:
                    #         writer.write('\n'.join(preds))
        t_loss = t_loss/t_steps
        logger.info("Training loss = %s", str(t_loss))
        results, eval_loss = evaluate(args, model, tokenizer, labels, 'validation')
        if results.get('f1') > best_f1_score and args.save_steps > 0:
            best_f1_score = results.get('f1')
            logger.info(" new Best F1 = %s", str(best_f1_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        elif results.get('f1') > best_f1_score and args.save_steps == 0:
            best_f1_score=results.get('f1')
            logger.info(" new Best F1 = %s", str(best_f1_score))
        else:
            logger.info("  Best F1 still at = %s", str(best_f1_score))
        logger.info("Validation loss = %s", str(eval_loss))
        train_losses.append(t_loss)
        val_losses.append(eval_loss)
        if args.wandb:
            wandb.log({'train_loss': t_loss})
            wandb.log({'val_loss': eval_loss})
            
        try:
            preds = evaluate(args, model, tokenizer, labels, mode="test")
            all_tests.append((results.get('f1'),preds))
            if args.wandb: 
                wandb.log({'all_tests_f1': results.get('f1')})
        except Exception as e:
            print(e)
        epnum+=1

            # Saving predictions
            # output_test_predictions_file = os.path.join(args.output_dir, "test_predictions_"+str(best_f1_score)+"_seed_"+str(args.seed)+"epoch_"+str(epnum -1)+".txt")
            # with open(output_test_predictions_file, "w") as writer:
            #     writer.write('\n'.join(preds))
            # tr_loss += loss.item()
    best_acc=0
    res=None
    print(len(all_tests))
    for i in all_tests:
        if i[0]>best_acc:
            best_acc=i[0]
            res=i[1]
    # fn=args.model_loc.split('/')[0]
    # fn=fn[15:]
    if res is not None:
        output_test_predictions_file = os.path.join(args.output_dir,args.save_file_start.replace('_','-')+"_" + str(best_acc)[:5]+"_seed_"+str(args.seed)+"_ep_"+str(args.num_train_epochs)+".txt")
        print(output_test_predictions_file)
        with open(output_test_predictions_file, "w+") as writer:
            writer.write('\n'.join(res))

    print("Train losses:",  train_losses)
    print("Val losses:", val_losses)
    print(epnum)
    
    # with open('to_plot2.pkl', 'wb') as f:
    #     pickle.dump([train_losses, val_losses, list(range(epnum))], f)
    # tr_loss += loss.item()
            # optimizer.step()
            # scheduler.step()
            # model.zero_grad()
            # global_step += 1

               

        # results = evaluate(args, model, tokenizer, labels, 'validation')

        

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, mode, prefix=""):

    eval_dataset = load_and_cache_examples(args, tokenizer, labels, mode=mode)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate)
    results = {}

    # Evaluation
    if mode=='validation':
        logger.info("***** Running validation %s *****", prefix)
    else:
        logger.info("***** Running test %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2]}
            '''print(inputs["input_ids"])
            print(inputs["attention_mask"])
            print(inputs["token_type_ids"])'''
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)
    if mode == "test":
        preds_list = []
        label_map = {i: label for i, label in enumerate(labels)}

        for i in range(out_label_ids.shape[0]):
            if eval_dataset[i][2] == 0:
                preds_list.append('not found')
            else:
                preds_list.append(label_map[preds[i]])

        return preds_list

    else:
        result = acc_and_f1(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        return results, eval_loss


class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, present=None):
        self.input_ids = input_ids
        self.labels = labels
        self.present = present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def collate(examples):
    padding_value = 0

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([t[1] for t in examples])

    return first_sentence_padded, first_sentence_attn_masks, labels


def load_and_cache_examples(args, tokenizer, labels, mode):

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(examples, labels, tokenizer, args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f['input_ids'] for f in features]
    all_labels = [f['label'] for f in features]
    args = [all_input_ids, all_labels]
    if 'present' in features[0]:
        present = [1 if f['present'] else 0 for f in features]
        args.append(present)

    dataset = CustomDataset(*args)
    return dataset


def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Optional Parameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_type", type=str,
                        default='bert', help='type of model xlm/xlm-roberta/bert')
    parser.add_argument("--model_name", default='bert-base-multilingual-cased',
                        type=str, help='name of pretrained model/path to checkpoint')
    parser.add_argument("--save_steps", type=int, default=1, help='set to -1 to not save model')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_pred", action="store_true", help="Whether to run zero-shot learning.")
    parser.add_argument("--logging_steps", type=int, default=500, help='set to -1 to not save model')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help='set to -1 to not save model')
    parser.add_argument("--max_seq_length", default=128, type=int, help="max seq length after tokenization")
    parser.add_argument(
        "--model_loc",
        default=None,
        type=str,
        required=False,
        help="Pretrained model location.",
    )
    parser.add_argument(
        "--save_file_start",
        default='',
        type=str,
        required=False,
        help="The file to write fi values.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="specify to use wandb logging"
    )
    parser.add_argument(
        "--experiment-name",
        default="en_hi_awesome_experiment",
        type=str,
        help="display name of the experiment on wandb"
    )

    args = parser.parse_args()

    # WANDB SETUP
    if args.wandb:
        wandb.init(project="MLMPretraining-SA", entity="csalt-pretraining", name=args.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    # Prepare data
    labels = get_labels(args.data_dir)
    num_labels = len(labels) # 0 for negative 1 for positive 2 for neutral
    # Initialize model
    tokenizer_class = {"xlm": XLMTokenizer, "bert": BertTokenizer, "xlm-roberta": XLMRobertaTokenizer, "residual-bert": BertTokenizer}
    config_class = {"xlm": XLMConfig, "bert": BertConfig, "xlm-roberta": XLMRobertaConfig, "residual-bert": BertConfig}
    model_class = {"xlm": XLMForSequenceClassification, "bert": BertForSequenceClassification, "xlm-roberta": XLMRobertaForSequenceClassification, "residual-bert": ResidualBertForSequenceClassification}

    if args.model_type.startswith('residual-bert'):
        parselist = args.model_type.split('_')
        assert len(parselist) == 3
        args.model_type = parselist[0]
        res_layer = int(parselist[1])
        res_dropout = float(parselist[2])

    if args.model_type not in tokenizer_class.keys():
        print("Model type has to be xlm/xlm-roberta/bert/residual-bert")
        exit(0)
    tokenizer = tokenizer_class[args.model_type].from_pretrained(args.model_name, do_lower_case=True)

    config=config_class[args.model_type].from_pretrained(
        # args.model_name,
        # 'ArchikiCombinedMLM-All-bert/checkpoint-18000/config.json',
        args.model_loc+'/config.json' if args.model_loc else args.model_name,
        output_hidden_states=True,
        num_labels=num_labels)
        # config=config_class[args.model_type].from_pretrained(args.model_name,num_labels=num_labels)
    print("config loaded")

    if args.model_type == 'residual-bert':
        model = model_class[args.model_type].from_pretrained(
            args.model_loc+'/pytorch_model.bin' if args.model_loc else args.model_name,
            config=config,
            res_layer=res_layer,
            res_dropout=res_dropout
        )
    else:
        model = model_class[args.model_type].from_pretrained(
            # args.model_name, 
            # num_labels=num_labels
            args.model_loc+'/pytorch_model.bin' if args.model_loc else args.model_name,
            # 'ArchikiCombinedMLM-All-bert/checkpoint-18000/pytorch_model.bin',
            config=config
        )
    print("model loaded")

    model.to(args.device)

    # Training

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, labels, mode="train")
        valid_dataset = load_and_cache_examples(
            args, tokenizer, labels, mode="validation")
        global_step, tr_loss = train(
            args, train_dataset, valid_dataset, model, tokenizer, labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation

    results = {}

    print("Loading best model for evaluation")
    config=config_class[args.model_type].from_pretrained(
        args.output_dir+'/config.json',
        num_labels=num_labels)
    if args.model_type == "residual-bert":
        model = model_class[args.model_type].from_pretrained(
            args.output_dir+'/pytorch_model.bin',
            config=config,
            res_layer=res_layer,
            res_dropout=res_dropout
        )
    else:
        model = model_class[args.model_type].from_pretrained(
            args.output_dir+'/pytorch_model.bin',
            config=config)
    print("model loaded")
    model.to(args.device)

    result, _ = evaluate(args, model, tokenizer, labels, mode="validation")
    best_acc = result.get('f1')
    if args.do_pred:
        best_acc = result.get('f1')
        res = evaluate(args, model, tokenizer, labels, mode="test")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if res is not None:
            output_test_predictions_file = os.path.join(args.output_dir,args.save_file_start.replace('_','-')+"_" + str(best_acc)[:5]+"_seed_"+str(args.seed)+"_ep_"+str(args.num_train_epochs)+".txt")
            print(output_test_predictions_file)
            with open(output_test_predictions_file, "w+") as writer:
                writer.write('\n'.join(res))
    # preds = evaluate(args, model, tokenizer, labels, mode="test")

    # # Saving predictions
    # output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    # with open(output_test_predictions_file, "w") as writer:
    #     writer.write('\n'.join(preds))

    return results


if __name__ == "__main__":
    main()
