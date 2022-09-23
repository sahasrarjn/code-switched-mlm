true_file = '../../pretraining/Data/MLM/withLIDtags/Hindi/combined/combined.txt'
pred_file = '../data/CSdata/emoji/combined-freqmlm-lidtags-processed.txt'

PRED = {
    "HI" : 0,
    "EN" : 0,
    "AMB-HI" : 0,
    "AMB-EN" : 0,
    "OTHER" : 0
}

CONFUSION_MAT = {
    "HI" : PRED.copy(),
    "EN" : PRED.copy(),
    "OTHER" : PRED.copy()
}

tf = open(true_file, 'r')
pf = open(pred_file, 'r')
tf_lines = tf.readlines()
pf_lines = pf.readlines()
tf.close()
pf.close()

for tfl, pfl in zip(tf_lines, pf_lines):
    tfl = tfl.strip().split('\t')
    pfl = pfl.strip().split('\t')
    if tfl[0].lower().strip() != pfl[0].lower().strip():
        print("#"*100 + '\n' + "MISMATCH: " + tfl[0] + " : "+ pfl[0] + '\n' + "#"*100)
    if len(tfl) == 1:
        if tfl[0] == '':
            continue
        else:
            print("#"*100 + '\n' + "ANOMALY: " + tfl[0] + " : " + pfl[0] + '\n' + "#"*100)
    # if tfl[1] != 'OTHER' and pfl[1] == 'OTHER':
    #     print(tfl[0], pfl[0], tfl[1], pfl[1])

    tfl[1] = 'EN' if tfl[1] == 'ENG' else tfl[1]
    pfl[1] = 'EN' if pfl[1] == 'ENG' else pfl[1]
    CONFUSION_MAT[tfl[1]][pfl[1]] += 1

print('TRUE\PRED', 'HI', 'AMB-HI', 'EN', 'AMB-EN', 'OTHER', sep='\t\t')
for tkey in ['HI', 'EN', 'OTHER']:
    print(tkey, end='\t\t\t')
    for pkey in ['HI', 'AMB-HI', 'EN', 'AMB-EN', 'OTHER']:
        print(CONFUSION_MAT[tkey][pkey], end='\t\t')
    print()

