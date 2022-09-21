with open("en_hi_freq.txt", 'r') as f:
	lno = 1
	count = 0
	for line in f.readlines():
		toks = len(line.strip().split('\t')[1].split(' '))
		if toks > 511:
			count += 1
			print(lno, ":", toks)
		lno += 1
print(count)

