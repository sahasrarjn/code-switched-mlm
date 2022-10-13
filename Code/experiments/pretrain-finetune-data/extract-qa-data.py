import argparse
from transformers.data.processors.squad import SquadV1Processor


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, default='../../../Data/Processed_Data/QA_EN_HI', help='Data dir')
parser.add_argument('-t', '--target', type=str, default='../../taggedData/Hindi/en_hi_qa_fine_baseline.txt', help='Target file')
parser.add_argument('-m', '--model_name_or_path', default='bert-base-multilingual-cased', type=str, help="Path to pre-trained model or shortcut name selected in the list: ",)
parser.add_argument(
    "--train_file",
    default='train-v2.0.json',
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--predict_file",
    default='dev-v2.0.json',
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
args = parser.parse_args()


def load_and_cache_examples(args, evaluate=False):
    # Load data features from cache or dataset file
    processor = SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
    else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
    return examples


train_dataset = load_and_cache_examples(args, evaluate=False)
eval_dataset = load_and_cache_examples(args, evaluate=True)

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        
with open(args.target, 'w+') as t:
    for data in train_dataset:
        t.write(data.question_text + '\n')
        t.write('\n'.join(tokenizer.tokenize(data.context_text)) + '\n')

    for data in eval_dataset:
        t.write(data.question_text + '\n')
        t.write('\n'.join(tokenizer.tokenize(data.context_text)) + '\n')