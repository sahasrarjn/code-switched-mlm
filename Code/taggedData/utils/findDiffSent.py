f1 = 'en_hi_freq.txt'
f2 = 'en_hi_switch.txt'

s1 = []
s2 = []

with open(f1, 'r') as f:
    lines = f.readlines()
    s1 = [line.rstrip().split('\t')[0] for line in lines]

with open(f2, 'r') as f:
    lines = f.readlines()
    s2 = [line.rstrip().split('\t')[0] for line in lines]

s1 = sorted(s1)
s2 = sorted(s2)

n, m = len(s1), len(s2)
i, j = 0, 0

while(i < n and j < m):
    if s1[i] == s2[j]:
        i += 1
        j += 1
    
