from tqdm import tqdm

mask_file = "/home/sahasra/pretraining/Code/taggedData/en_hi_switch.txt"
out_file = "probe-target-switch-points.txt"

ofp = open(out_file, 'w+')
with open(mask_file, 'r') as fp:
    for line in tqdm(fp.readlines(), desc="Lines"):
        line = line.strip()
        sentence, masks = line.split('\t')
        mask_list = masks.strip().split(' ')
        probe_target = [int(m=='MASK') for m in mask_list] # 1 for MASK, 0 for NOMASK
        ofp.write(sentence + '\t' + ' '.join([str(x) for x in probe_target]) + '\n')
ofp.close()
