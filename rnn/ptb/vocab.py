import codecs
import collections
from operator import itemgetter
#http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

raw_data = "/Users/yxd/Downloads/simple-examples/data/ptb.train.txt"
vocab = "/Users/yxd/Downloads/simple-examples/data/vocab"

counter = collections.Counter()
with codecs.open(raw_data, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
sorted_word = sorted(counter.items(),key=itemgetter(1),reverse=True)
sorted_word = [e[0] for e in sorted_word]
sorted_word = ['<eos>'] + sorted_word
print(len(sorted_word))
print(sorted_word[0],sorted_word[-1])
if len(sorted_word) > 10000:
    sorted_word = sorted_word[:10000]
with codecs.open(vocab, 'w', 'utf-8') as f:
    for word in sorted_word:
        f.write(word + "\n")