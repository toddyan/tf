import tensorflow as tf
import sys
sys.path.append("../LeNet5")
from comm import ConvLayer
import infer
class Conf:
    def __init__(self):
        # structure
        self.image_shape  = (28,28,3)
        self.classes      = 10
        self.conv_layers  = [ConvLayer((5,5,3,8), (1,1,1,1), (1,2,2,1), (1,2,2,1)),
                             ConvLayer((5,5,8,16),(1,1,1,1),(1,2,2,1),(1,2,2,1))]
        self.fc_layers    = [128, self.classes]


def parse(record):
    example = tf.parse_single_example(
        record,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
            "weight": tf.FixedLenFeature([], tf.int64),
            "height": tf.FixedLenFeature([], tf.int64),
            "channel": tf.FixedLenFeature([], tf.int64)
        }
    )
    image, label = example["image"], example["label"]
    height, weight, channel = example["height"], example["weight"], example["channel"]
    image_shape = tf.stack([height, weight, channel])
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, shape=image_shape)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    return image, label


train_files = tf.train.match_filenames_once("E:/tmp/mnist_tfrecord")
target_height = 28
target_weight = 28
batch_size = 64
shuffle_buffer = 1000
epochs = 10
learning_rate = 0.0001

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parse)
dataset = dataset.map(lambda img, lab: (tf.image.resize_images(img,(target_height,target_weight)),lab))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).repeat(epochs)
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

logits = infer.infer(Conf(), image_batch, True, None)
cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batch)
)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), label_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as s:
    s.run([tf.global_variables_initializer(),
           tf.local_variables_initializer(),])
    s.run(iterator.initializer)
    step = 1
    while True:
        try:
            _, acc = s.run([optimizer, accuracy])
            if step%10 == 0:
                print(step, "acc:", acc)
                saver.save(sess=s,save_path="E:/tmp/mnist/model.ckpt",global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break