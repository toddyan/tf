import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def distort(image, order):
    if order == 0:
        image = tf.image.random_brightness(image, max_delta= 32.0/255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    if order == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta= 32.0/255.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, weight, bbox, enable_hir_flip):
    if bbox == None:
        bbox = tf.constant([[[0.0,0.0,1.0,1.0]]], dtype=tf.float32)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(image_size=tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image, size=(height, weight), method=np.random.randint(low=0,high=4))
    if enable_hir_flip == True:
        distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort(distorted_image, np.random.randint(low=0,high=2))
    return distorted_image

image_raw = tf.gfile.FastGFile("/Users/yxd/Downloads/flower_photos/roses/12240303_80d87f77a3_n.jpg",'r').read()
with tf.Session() as s:
    image = tf.image.decode_jpeg(image_raw)
    boxex = tf.constant([[[0.05,0.05,0.9,0.7],[0.2,0.2,0.8,0.86]]])
    for i in range(5):
        result = preprocess_for_train(image,299,299,boxex,True)
        result = result.eval()
        print(result.shape)
        plt.imshow(result)
        plt.show()


