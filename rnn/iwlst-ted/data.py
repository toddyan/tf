import codecs
import collections
from operator import itemgetter

# https://wit3.fbk.eu/mt.php?release=2015-01
# https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
# git clone -n https://github.com/moses-smt/mosesdecoder.git --depth 1
# git checkout HEAD scripts
# ./mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l en <train.tags.en-zh.en >train.txt.en
# sed 's/ //g; s/\B/ /g; s/[,.?;:\'"!@#$%^&*\(\)_+=\[\]{}<>-]/ /g; s/，/ ， /g; s/。/ 。 /g; s/？/ ？ /g; s/：/ ： /g; s/“/ “ /g; s/”/ ” /g; s/（/ （ /g; s/）/ ） /g ' ./train.tags.en-zh.zh >train.txt.zh

DATA_ROOT="E:/Download/en-zh/"
EN_VOCAB_SIZE = 10000
ZH_VOCAB_SIZE = 4000

seperated_file_path = DATA_ROOT + "train.txt.zh"
seperated_file = codecs.open(seperated_file_path,'w','utf-8')
with codecs.open(DATA_ROOT+"train.tags.en-zh.zh",'r','utf-8') as f:
    for line in f:
        line = line.strip()
        s = ' '.join(line)
        seperated_file.write(s + "\n")
seperated_file.close()

sep_datas = [DATA_ROOT+"train.txt.en",DATA_ROOT+"train.txt.zh"]
vocab_datas = [DATA_ROOT+"train.vocab.en",DATA_ROOT+"train.vocab.zh"]
for (data_file,vocab_file) in zip(sep_datas, vocab_datas):
    print(data_file,vocab_file)
    counter = collections.Counter()
    with codecs.open(data_file, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1
    sorted_word = sorted(counter.items(),key=itemgetter(1),reverse=True)
    sorted_word = [e[0] for e in sorted_word]
    sorted_word = ['<sos>','<eos>'] + sorted_word
    print(len(sorted_word))
    print(sorted_word[0],sorted_word[-1])
    if len(sorted_word) > 10000:
        sorted_word = sorted_word[:10000]
    with codecs.open(vocab_file, 'w', 'utf-8') as f:
        for word in sorted_word:
            f.write(word + "\n")