# coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt

train_images_file = 'MNIST_data/train-images.idx3-ubyte' # 训练集文件
train_labels_file = 'MNIST_data/train-labels.idx1-ubyte' # 训练集标签文件
test_images_file = 'MNIST_data/t10k-images.idx3-ubyte' # 测试集文件
test_labels_file = 'MNIST_data/t10k-labels.idx1-ubyte' # 测试集标签文件

def load_images(idx3_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为文件头魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('文件头魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print("指针位置（即偏移位置offset）指向:", offset)
    fmt_image = '>' + str(image_size) + 'B'
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure() 绘图函数

    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print("指针位置:", offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i]
        offset += struct.calcsize(fmt_image)
    return images


def load_labels(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为文件头魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('文件头魔数:%d, 标签数量(即图片数量): %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


if __name__ == '__main__':
    train_images = load_images(train_images_file)
    train_labels = load_labels(train_labels_file)
    # test_images = load_images(test_images_file)
    # test_labels = load_labels(test_labels_file)

    # 查看前十个数据及其标签以读取是否正确
    print("\n解析测试（查看前十个照片解析是否正确）")
    '''
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.pause(0.000001)
        plt.show()
    print('查看完成')
    '''

    # 多图合一显示,可以确实区间显示
    for i in range(10):
        plt.subplot(2, 5, 1 + i)
        plt.title(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
    plt.show()
    print('查看完成')
