import re
import csv
import glob
from collections import defaultdict

vocab = defaultdict(int)
totalCount = 0
fileCount = 0
DIR_LOC = 'supervised/'
files = glob.glob(DIR_LOC + '*/opus.*.en')

for file in files:
    with open(file, 'r', encoding='utf-8') as fo:
        for line in fo:
            sentence = line.strip()
            words = re.findall('[a-zA-Z]+', sentence)
            for word in words:
                if word.isnumeric():
                    print("#"*100)
                    print(word, " : ANOMALY")
                    print("#"*100)
                    continue
            
                word = word.lower()
                vocab[word] += 1
                totalCount += 1
    fileCount += 1
    print(f"[{fileCount}/{len(files)}] " + file + " DONE!!!")

sorted(vocab.items(), lambda kv: kv[1], reverse=True)

with open('opus-100-corpus-freq.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['word', 'count'])
    for k, v in vocab.items():
        writer.writerow([k, v])
