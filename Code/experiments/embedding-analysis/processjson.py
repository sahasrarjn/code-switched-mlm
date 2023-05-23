import json
from collections import defaultdict
from indictrans import Transliterator

with open('diverse-CS.json') as f:
    data = json.load(f)

trn = Transliterator(source='hin', target='eng', build_lookup=True)

map = defaultdict(list)
for cs in data:
    hi_input = trn.transform(cs['hi_input'])
    cs_output = trn.transform(cs['cs_output'])
    map[hi_input].append(cs_output)

json.dump(map, open('processedCS.json', 'w', encoding='utf-8'), ensure_ascii=False)