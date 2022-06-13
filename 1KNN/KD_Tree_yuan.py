import struct
from collections import Counter
from array import array as pyarray
from pylab import *  # 支持中文
from sklearn.neighbors import KDTree

# import warnings
# warnings.filterwarnings("ignore",category=DeprecationWarning)
train_images_file = 'MNIST_data/train-images.idx3-ubyte'  # 训练集文件
train_labels_file = 'MNIST_data/train-labels.idx1-ubyte'  # 训练集标签文件
test_images_file = 'MNIST_data/t10k-images.idx3-ubyte'  # 测试集文件
test_labels_file = 'MNIST_data/t10k-labels.idx1-ubyte'  # 测试集标签文件
# 读取数据
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

if __name__ == "__main__":
    train_img, train_label = load_mnist(train_images_file, train_labels_file)
    test_img, test_label = load_mnist(test_images_file, test_labels_file)


    #构造KDtree树
    t_before = time.time()
    print("构造KDtree中，请稍微！")
    kd_tree = KDTree(train_img)
    t_after = time.time()
    t_training = t_after - t_before
    print("构建KD树需要多少(多少秒): ", t_training)

    ## 得到测试集的预测结果，k表示几个近邻
    t_before = time.time()
    k = 1
    print("测试集测试中，请稍微！\nk值为：",k)
    test_neighbors = np.squeeze(kd_tree.query(test_img, k, return_distance=False))
    kd_tree_predictions = train_label[test_neighbors]
    t_after = time.time()

    ##完成测试需要的时间
    t_testing = t_after - t_before
    print("完成测试需要的时间(多少秒): ", t_testing)

    #输出准确度
    print("test_label的长度：", len(test_label))
    acc = np.sum(kd_tree_predictions == test_label) / len(test_label)
    print("准确度：", acc)
