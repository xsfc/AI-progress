import struct
from collections import Counter
from array import array as pyarray
from pylab import *  # 支持中文
from sklearn.neighbors import KDTree
mpl.rcParams['font.sans-serif'] = ['SimHei']
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

def Kdtree_create(train_img):
    ## 构造KD树
    t_before = time.time()
    print("构造KDtree中，请稍微！")
    kd_tree = KDTree(train_img)
    t_after = time.time()
    t_training = t_after - t_before
    print("构建KD树需要多少(多少秒): ", t_training)
    return kd_tree

def preds(test_img,k,num):
    print("测试集测试中，请稍微！\nk值为：", k)
    test_neighbors = np.squeeze(kd_tree.query(test_img, k, return_distance=False))
    kd_tree_predictions = train_label[test_neighbors] #[]是列表的意思
    kd_tree_prediction = np.array(kd_tree_predictions).reshape(num,k)
    # Counter()函数返回数组中元素的个数，most_common(1)限定最多的那个[(4, 5), (3, 3)]中的[(4,5)]，[0][0]返回第几个数组的第几个数据
    kd_prediction = [Counter(kd_tree_prediction[i]).most_common(1)[0][0] for i in range(num)]
    kd_predictions = np.array(kd_prediction).reshape(num,1)
    acc = np.sum(kd_predictions == test_label) / len(test_label)
    return acc

if __name__ == "__main__":
    t_before = time.time()
    train_img, train_label = load_mnist(train_images_file, train_labels_file)
    test_img, test_label = load_mnist(test_images_file, test_labels_file)
    num = 300
    test_img = test_img[:num]
    test_label = test_label[:num]

    kd_tree = Kdtree_create(train_img)
    '''
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

    print("test_label的长度：", len(test_label))
    acc = np.sum(kd_tree_predictions == test_label) / len(test_label)
    print("准确度：", acc)
    '''
    k_acc = []
    j = 0
    for k in range(1, 100, 2):
        acc = preds(test_img,k,num)
        k_acc.append(acc)
        print("准确度：", k_acc[j],'\n')
        j = j + 1

    t_after = time.time()
    t = t_after - t_before
    print("耗时(多少秒): ", t)

    list = [x for x in range(1, 100, 2)]
    x = range(len(list))
    y = k_acc
    plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'K值与准确率曲线图')
    plt.legend()  # 让图例生效
    plt.xticks(x, list, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("K值")  # X轴标签
    plt.ylabel("准确率")  # Y轴标签
    plt.title("K值影响表")  # 标题
    plt.show()
