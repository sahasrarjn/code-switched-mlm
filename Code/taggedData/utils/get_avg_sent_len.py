import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='../Spanish/en_es_switch.txt', help='Input file to split into train, eval files')
args = parser.parse_args()

counts = []
with open(args.file, 'r') as fp:
    for line in fp.readlines():
        count = len(line.split('\t')[0].strip().split())
        counts += [count]

plt.hist(counts)
plt.savefig("im")
plt.show()

print(f"Average Sentence Length: {np.mean(counts)}")
print(f"Stddev Sentence Length: {np.std(counts)}")
