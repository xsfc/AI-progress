import struct, os
from array import array as pyarray
import numpy as np
from matplotlib import pyplot as plt
from numpy import append, array, int8, uint8, zeros

# 训练集文件
train_images_file = 'MNIST_data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_file = 'MNIST_data/train-labels.idx1-ubyte'
# 测试集文件
test_images_file = 'MNIST_data/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_file = 'MNIST_data/t10k-labels.idx1-ubyte'


def load_mnist(fname_image, fname_label):
    digits = np.arange(10)

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(N):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((1, rows * cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def show_image(imgdata, imgtarget, show_column, show_row):
    # 注意这里的show_column*show_row==len(imgdata)
    for index, (im, it) in enumerate(list(zip(imgdata, imgtarget))):
        xx = im.reshape(28, 28)
        plt.subplots_adjust(left=1, bottom=None, right=3, top=4, wspace=None, hspace=None)
        plt.subplot(show_row, show_column, index + 1)
        plt.axis('off')
        plt.imshow(xx, cmap='gray', interpolation='nearest')
        plt.title('target:%i' % it)


if __name__ == "__main__":
    train_image, train_label = load_mnist(train_images_file, train_labels_file)
    test_image, test_label = load_mnist(test_images_file, test_labels_file)
    # print(train_image[1])
    print("AAAA")
    images = np.array(train_image[1].reshape(28, 28))
    plt.imshow(images, 'gray')
    plt.show()
    print(train_label[1])
