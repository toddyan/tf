import tensorflow as tf
import time
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


test_files  = tf.train.match_filenames_once("E:/tmp/mnist_tfrecord-test")
batch_size = 10000
target_height = 28
target_weight = 28

########## test set #############
testset = tf.data.TFRecordDataset(test_files)
testset = testset.map(parse)
testset = testset.map(lambda img,lab:(tf.image.resize_images(img,(target_height,target_weight)),lab))
testset = testset.batch(batch_size).repeat(10000)
test_iterator = testset.make_initializable_iterator()
test_image_batch, test_label_batch = test_iterator.get_next()
test_logits = infer.infer(Conf(), test_image_batch, False, None)
correct_prediction = tf.equal(tf.argmax(test_logits,1), test_label_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver() # no name map used

with tf.Session() as s:
    s.run([tf.global_variables_initializer(),
           tf.local_variables_initializer()])
    s.run(test_iterator.initializer)
    while True:
        try:
            ckpt = tf.train.get_checkpoint_state("E:/tmp/mnist")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(s, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                acc = s.run(accuracy)
                print("epoch[%s] validation acc=%f" % (global_step, acc))
        except tf.errors.OutOfRangeError:
            break
        time.sleep(5)

