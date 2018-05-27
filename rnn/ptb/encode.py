import codecs
import sys

def get_word_code(word, dict):
    return str(dict[word]) if word in dict else str(dict['<unk>'])

raw_data = "/Users/yxd/Downloads/simple-examples/data/ptb.train.txt"
vocab = "/Users/yxd/Downloads/simple-examples/data/vocab"
out_data = "/Users/yxd/Downloads/simple-examples/data/ptb.train.code"


with codecs.open(vocab, 'r', 'utf-8') as f:
    vocab = [w.strip() for w in f.readlines()]
dict = {k:v for (k,v) in zip(vocab,range(len(vocab)))}

fin = codecs.open(raw_data, 'r', 'utf-8')
fout = codecs.open(out_data, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']
    fout.write(' '.join([get_word_code(w,dict) for w in words]) + '\n')
fin.close()
fout.close()