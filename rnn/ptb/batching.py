import numpy as np
import tensorflow as tf
import codecs
training_path = "/Users/yxd/Downloads/simple-examples/data/ptb.train.code"
batch_size = 4
timestep = 35

def read_data(file):
    with codecs.open(training_path, 'r', 'utf-8') as f:
        total = ' '.join([line.strip() for line in f.readlines()])
    return [int(w) for w in total.split()]

def make_batch(id_list, batch_size, timestep):
    num_batches = (len(id_list)-1) // (batch_size*timestep)
    data = np.reshape(
        np.array(id_list[:num_batches*batch_size*timestep]),
        [batch_size, num_batches*timestep]
    )
    # split [batch_size, num_batches*timestep] into num_batches * [batch_size, timestep]
    data_batches = np.split(data, num_batches, axis=1)

    label = np.reshape(
        np.array(id_list[1:num_batches*batch_size*timestep+1]),
        [batch_size, num_batches*timestep]
    )
    label_batches = np.split(label, num_batches, axis=1)
    return list(zip(data_batches,label_batches))

def main():
    train_batches = make_batch(read_data(training_path),batch_size,timestep)
    print(len(train_batches))
    print(train_batches[0])

if __name__ == "__main__":
    main()