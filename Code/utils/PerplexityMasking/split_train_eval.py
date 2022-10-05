import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input_file', type=str, required=True, help='Input file to split into train, eval files')
parser.add_argument('-t', '--train_file', type=str, required=True, help='Output train file')
parser.add_argument('-e', '--eval_file', type=str, required=True, help='Output train file')
parser.add_argument('--frac', type=float, default=0.1, help='Split fraction for eval/train')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)

with open(args.input_file, 'r') as f:
    lines = f.readlines()
    numTotal = len(lines)
    numEval = int(args.frac * numTotal)
    evalIdx = random.sample(range(numTotal), numEval)

    print(f"Split: {args.train_file} ({numTotal-numEval}), {args.eval_file} ({numEval})")
    with open(args.train_file, 'w+') as tf:
        with open(args.eval_file, 'w+') as ef:
            for i, line in enumerate(tqdm(lines)):
                if i in evalIdx:
                    ef.write(line)
                else:
                    tf.write(line)