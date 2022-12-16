import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, default='../Spanish/all.txt', help="Source file")
parser.add_argument('-t', '--target', type=str, default='../Spanish/all_clean.txt', help="Target file")
args = parser.parse_args()


def remove_emojis(data):
    emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u2100-\u23ff"
            u"\u200d"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
        "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def remove_special_chars(data):
    special_char = re.compile(r'[@_!#$%.;`^&,*=()<>?/\'’´¿…\|"“”}{~:-]')
    data = re.sub(special_char, ' ', data)
    data = re.sub(r'\[', '', data)
    data = re.sub(r'\]', '', data)
    return data


def clean_line(line):
    line = remove_emojis(line)
    # line = remove_special_chars(line)
    line = ' '.join(line.split())
    return line + '\n'


with open(args.target, 'w+') as out:
    with open(args.source, 'r') as inf:
        lines = inf.readlines()
        for line in tqdm(lines):
            line = clean_line(line)
            if line != '':
                out.write(line)