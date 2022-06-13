import struct, os
from array import array as pyarray
from pylab import *  # 支持中文
from aKNN import Knn
import time
time_start = time.clock()  # 记录开始时间

train_images_file = 'MNIST_data/train-images.idx3-ubyte' # 训练集文件
train_labels_file = 'MNIST_data/train-labels.idx1-ubyte' # 训练集标签文件
test_images_file = 'MNIST_data/t10k-images.idx3-ubyte' # 测试集文件
test_labels_file = 'MNIST_data/t10k-labels.idx1-ubyte' # 测试集标签文件

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

    #调用Knn算法预测,默认K数值为5,train_img,train_label,test_img,test_label
    Use_Knn(5,train_img,train_label,test_img,test_label)
    print("此时的K值为：5")
    print('\n\n\n\n\n')

    time_end = time.clock()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)