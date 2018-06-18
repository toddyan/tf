import sys
sys.path.append("..")
import globalconf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)/255.0
X_test = X_test.astype(np.float32)/255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)

def dataset_to_sprite(images):
    if isinstance(images, list):
        images = np.array(images)
    h = images.shape[1]
    w = images.shape[2]
    m = int(np.ceil(np.sqrt(images.shape[0])))
    sprite = np.ones(shape=[h*m,w*m],dtype=np.float32)
    for i in range(m):
        for j in range(m):
            index = i * m + j
            if index < images.shape[0]:
                sprite[i*h:(i+1)*h,j*w:(j+1)*w] = images[index]
    return sprite

sprite_image = dataset_to_sprite(1.0-X_test)
sprite_path = globalconf.get_root() + "tensorboard/visual/sprite.png"
meta_path = globalconf.get_root() + "tensorboard/visual/meta.tsv"
plt.imsave(sprite_path,sprite_image,cmap='gray')
# plt.imshow(sprite_image, cmap='gray')
# plt.show()
with open(meta_path,'w') as f:
    f.write("Index\tLabel\n")
    for index in range(Y_test.shape[0]):
        f.write("%d\t%d\n"%(index, Y_test[index]))
