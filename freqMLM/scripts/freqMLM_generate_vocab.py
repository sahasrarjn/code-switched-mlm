'''
Generate vocabulary file from OPUS dataset

** Need to setup the data path correctly for the prepublished datasets. We cannot provide access to those dataset.
'''

import re
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def is_float(element) -> bool:
    try:
        float(element)
        return True
    except:
        return False

class Preprocess:
    def __init__(
        self, 

        # Monolingual data path
        datafile1_l1, 
        datafile1_l2, # dakshina
        # datafile2_l2, # hinglishnorm
        datafile2_l2  # samanantar
    ):
        self.vocab1 = self.get_unigram(datafile=datafile1_l1)
        # replace this csv thing with a proper english dataset
        
        # self.vocab1 = self.get_wms_data(datadir=datafile1_l1)
        self.vocab2 = self.get_vocab(datafile1=datafile1_l2, datafile2=datafile2_l2)
        # self.read_hinglish_sentences(datafile=datafile2_l2)

        self.show_vocab(self.vocab1, 'out/vocab_EN.txt', 'out/vocab_EN.png')
        self.show_vocab(self.vocab2, 'out/vocab_HI.txt', 'out/vocab_HI.png')
        
    
    def get_unigram(self, datafile):
        vocab = defaultdict(float)
        totalCount = 0
        with open(datafile, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                totalCount += int(row[1])
                vocab[row[0]] = float(row[1])
            
        for _, word in enumerate(vocab):
            vocab[word] = -np.log(vocab[word]) + np.log(totalCount)
        
        print("ENGLISH FREQ CSV Processed!!!")
        return vocab


    def get_vocab(self, datafile1, datafile2):
        '''Read lexicons and return vocab for the language (Dakshina Dataset v1.0)
            datafile1: Dakshina
            datafile2: Samanantar
        '''
        vocab = defaultdict(float)
        totalCount = 0
        print("HINDI Started!!!")

        with open(datafile1, "r") as file:
            for line in file:
                sentence = line.strip().split('\t')[1]
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
        print("DATAFILE 1 DONE!!!")

        with open(datafile2, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                words = re.findall('[a-zA-Z]+', line)
                for word in words:
                    if word.isnumeric():
                        print("#"*100)
                        print(word, " : ANOMALY")
                        print("#"*100)
                        continue
                    word = word.lower()
                    vocab[word] += 1
                    totalCount += 1
        print("DATAFILE 2 DONE!!!")

        # with open(datafile3, "r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         words = re.findall('[a-zA-Z]+', line)
        #         for word in words:
        #             if word.isnumeric():
        #                 print("#"*100)
        #                 print(word, " : ANOMALY")
        #                 print("#"*100)
        #                 continue
        #             word = word.lower()
        #             vocab[word] += 1
        #             totalCount += 1
        # print("DATAFILE 3 DONE!!!")

        for _, word in enumerate(vocab):
            vocab[word] = -np.log(vocab[word]) + np.log(totalCount)

        return vocab


    def show_vocab(self, vocab, filename, imagefile):
        og_stdout = sys.stdout
        with open(filename, 'w+') as outfile:
            sys.stdout = outfile
            for word, nll in sorted(vocab.items(), key=lambda kv: kv[1]):
                print(word, nll)
        sys.stdout = og_stdout

        distri = [v for _, v in vocab.items()]
        distri = sorted(distri)
        plt.hist(distri)
        plt.ylabel('Count')
        plt.xlabel('Frequency (-ve log likelihood)')
        plt.savefig(imagefile)
        plt.clf()




if __name__ == '__main__':
    preprocess = Preprocess(
        '../data/english_unigram_freq.csv', 
        # '../data/wms/', 

        # generated from OPUS 100 using script create_vocab_opus_100.py
        # '../data/opus-100-corpus-freq.csv', 

        '../data/dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.tsv', 
        # '../data/hinglishNorm/dataset/sentences.txt',
        '../data/Samanantar/train_romanized.hi'
    )