import struct, os
from array import array as pyarray
import numpy as np
from matplotlib import pyplot as plt
from numpy import append, array, int8, uint8, zeros
from aKNN import Knn

# 训练集文件
train_images_file = 'MNIST_data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_file = 'MNIST_data/train-labels.idx1-ubyte'
# 测试集文件
test_images_file = 'MNIST_data/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_file = 'MNIST_data/t10k-labels.idx1-ubyte'

#读取数据
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

#显示照片
def show_image(imgdata, imgtarget, show_column, show_row):
    # 注意这里的show_column*show_row==len(imgdata)
    for index, (im, it) in enumerate(list(zip(imgdata, imgtarget))):
        xx = im.reshape(28, 28)
        plt.subplot(show_row, show_column, index + 1)
        plt.axis('off')
        plt.imshow(xx, cmap='gray', interpolation='nearest')
        plt.title('target:%i' % it)

 # 归一化进程
    # 图片的存储方式为灰度图片，大小为28*28，
    # 每个像素的取值范围在0到255，如果直接运算会导致运算量过大，
    # 再结合其为灰度图片，我将其数值进行一定处理，简化后续的计算。
    # 处理的方式为若像素值大于127,，则赋值为1，否则赋值为0
def Data_ErZhiHua(data):
    '''放入主函数测试
    X_train_1 = Data_ErZhiHua(X_train[:5])
    for i in range(5):
        print(X_train_1[i])
    '''
    print("数据二值化进程中！")
    row = data.shape[0]  # 60000
    col = data.shape[1]  # 784
    for i in range(row):
        for j in range(col):
            if data[i][j] > 127:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data

#Knn算法实现
def Use_Knn(k,train_img,train_label,test_img,test_label):
    #K值,
    knn = Knn(k)
    #训练，KNN算法不需要训练，只是一个传递数据集的作用
    knn.fit(train_img, train_label)
    print("选择训练的数据集大小为：",train_img.shape[0])

    #用test_img,test_label数据集预测
    print("预测数据集大小为：", test_label.shape[0])
    preds = knn.predict(test_img,test_label)

    #输出的结果集为preds
    # 准确度分析
    print("test_label的长度：",len(test_label))
    acc = np.sum(preds == test_label) / len(test_label)
    print("准确度：",acc)


if __name__ == "__main__":
    train_img, train_label = load_mnist(train_images_file, train_labels_file)
    test_img, test_label = load_mnist(test_images_file, test_labels_file)
    '''输出照片测试
    print(X_train[1])
    print("AAAA")
    images = np.array(X_train[1].reshape(28, 28))
    plt.imshow(images, 'gray')
    plt.show()
    print(X_label[1])
    '''

    '''测试训练集为10000，测试集为2
    train_num=10000
    test_num=2
    '''

    '''测试训练集为20000，测试集为20'''
    train_num=20000
    test_num=20



    train_img=Data_ErZhiHua(train_img[:train_num])
    test_img=Data_ErZhiHua(test_img[:test_num])
    print("数据二进制进程完成！")

    #调用Knn算法预测,默认K数值为5,train_img,train_label,test_img,test_label
    Use_Knn(5,train_img,train_label[:train_num],test_img,test_label[:test_num])
    print("此时的K值为：5")
    print('\n\n\n\n\n')


    #调用函数进行图像的展示，9=3*3=1*9
    test_img = test_img[:9]
    test_label = test_label[:9]
    show_image(test_img, test_label, 3, 3)


    for i in [3,7,9]:
        Use_Knn(i,train_img,train_label[:train_num],test_img,test_label[:test_num])
        print("此时的K值为：",i)

