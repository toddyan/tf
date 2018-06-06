import codecs

# DATA_ROOT="E:/Download/en-zh/"
DATA_ROOT="/Users/yxd/Downloads/en-zh/"

def get_word_code(word, dict):
    return str(dict[word]) if word in dict else str(dict['<unk>'])


tasks = [(DATA_ROOT+"train.txt.en", DATA_ROOT+"train.vocab.en", DATA_ROOT+"train.code.en"),
         (DATA_ROOT+"train.txt.zh", DATA_ROOT+"train.vocab.zh", DATA_ROOT+"train.code.zh")]

for (raw_data,vocab,out_data) in tasks:
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
