'''
Generate LID tags for EN-ES language pair code switched sentences

Input args:
    -s, --source: Source file
    -t, --target: Target file
'''

import argparse
from codeswitch.codeswitch import LanguageIdentification

lidIden = LanguageIdentification('spa-eng') 

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../Spanish/all_clean.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='../Spanish/all_switch_lid.txt', help="Target file")
args = parser.parse_args()

def word_lid(token_lids):
    sp_count = 0
    for lid in token_lids:
        if lid == 'ES': sp_count += 1
    if 2*sp_count >= len(token_lids):
        return 'ES'
    else:
        return 'EN'

with open(args.target, 'w+') as fo:
    with open(args.source, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            line = line.strip()
            result = lidIden.identify(line)

            lids = []
            token_lids = []
            
            for res in result[1:-1]:
                token = res['word']
                lid = 'ES' if res['entity'] == 'spa' else 'EN'

                if token[:2] == '##':
                    token_lids.append(lid)
                else:
                    if token_lids != []:
                        lids.append(word_lid(token_lids))
                    token_lids = [lid]
            
            if token_lids != []:
                lids.append(word_lid(token_lids))
            
            words = line.split()
            for word, lid in zip(words, lids):
                fo.write(word + '\t' + lid + '\n')
            fo.write('\n')
