import os 

directory = '../data/CS_LID_gluecos'

def print_sentences(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        buffer = []
        for line in lines:
            print(line)
            if line == '':
                print(buffer)
                continue
            word, lid = line.split('\t')
            buffer.append((word, lid))
    

print_sentences(directory + '/train.txt')

# for filename in os.listdir(directory):
    # f = os.path.join(directory, filename)
    # if os.path.isfile(f):
    #     print_sentences(f)
    # break