import os 

filen = '../../Data/MLM/Hindi/L3Cube-HingLID/processed/l3cube_140k_44k_tagged.txt'

def print_sentences(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        buffer = ""
        for line in lines:
            line = line.strip()
            if line == '':
                print(buffer)
                buffer = ""
                continue
            word, lid = line.split('\t')
            buffer += word + ' '
    
print_sentences(filen)

# for filename in os.listdir(directory):
    # f = os.path.join(directory, filename)
    # if os.path.isfile(f):
    #     print_sentences(f)
    # break