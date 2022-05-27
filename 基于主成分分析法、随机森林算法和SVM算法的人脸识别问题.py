from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score
import numpy as np
import cv2
import os
import math
import random

CUT_X = 8
CUT_Y = 4

# 为了让LBP具有旋转不变性，将二进制串进行旋转。
# 假设一开始得到的LBP特征为10010000，那么将这个二进制特征，
# 按照顺时针方向旋转，可以转化为00001001的形式，这样得到的LBP值是最小的。
# 无论图像怎么旋转，对点提取的二进制特征的最小值是不变的，
# 用最小值作为提取的LBP特征，这样LBP就是旋转不变的了。
def minBinary(pixel):
    length = len(pixel)
    zero = ''
    # range(length)[::-1] 使得i从01234变为43210
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'


def LBP(FaceMat, R=2, P=8):
    pi = math.pi
    LBPoperator = np.mat(np.zeros([np.shape(FaceMat)[0], np.shape(FaceMat)[1] * np.shape(FaceMat)[2]]))
    for i in range(np.shape(FaceMat)[0]):
        # 对每个图像进行处理
        face = FaceMat[i, :]
        W, H = np.shape(face)
        tempface = np.mat(np.zeros((W, H)))
        for x in range(R, W - R):
            for y in range(R, H - R):
                repixel = ''
                pixel = int(face[x, y])
                # 圆形LBP算子
                for p in [2, 1, 0, 7, 6, 5, 4, 3]:
                    p = float(p)
                    xp = x + R * np.cos(2 * pi * (p / P))
                    yp = y - R * np.sin(2 * pi * (p / P))
                    xp = int(xp)
                    yp = int(yp)
                    if face[xp, yp] > pixel:
                        repixel += '1'
                    else:
                        repixel += '0'
                # minBinary保持LBP算子旋转不变
                tempface[x, y] = int(minBinary(repixel), base=2)
        # face_img = Image.fromarray(np.uint8(face))
        # face_img.show()
        # lbp_img = Image.fromarray(np.uint8(tempface))
        # lbp_img.show()
        LBPoperator[i, :] = tempface.flatten()
    return LBPoperator.T


def load_data():
    ORL_PATH = 'orl'
    train_X = []  # 训练集
    train_y = []
    test_X = []  # 测试集
    test_y = []
    person_dirnames = os.listdir(ORL_PATH)
    for dirname in person_dirnames:
        for i in range(1, 9):
            pic_path = os.path.join(ORL_PATH, dirname, str(i) + '.pgm')
            im = np.array(Image.open(pic_path).convert("L"))  # 读取文件并转化为灰度图
            train_X.append(im)
            train_y.append(int(dirname[1:]) - 1)
        for i in range(9, 11):
            pic_path = os.path.join(ORL_PATH, dirname, str(i) + '.pgm')
            im = np.array(Image.open(pic_path).convert("L"))  # 读取文件并转化为灰度图
            test_X.append(im)
            test_y.append(int(dirname[1:]) - 1)
    # 同时打乱X和y数据集。
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_X)
    random.seed(randnum)
    random.shuffle(train_y)
    print("训练集大小为: {}, 测试集大小为: {}".format(len(train_X), len(test_X)))
    return np.array(train_X), np.array(train_y).T, np.array(test_X), np.array(test_y).T


# 统计直方图
def calHistogram(ImgLBPope, h_num=CUT_X, w_num=CUT_Y):
    # 112 = 14 * 8, 92 = 23 * 4
    Img = ImgLBPope.reshape(112, 92)
    H, w = np.shape(Img)
    # 把图像分为8 * 4份
    Histogram = np.mat(np.zeros((256, h_num * w_num)))
    maskx, masky = H / h_num, w / w_num
    for i in range(h_num):
        for j in range(w_num):
            # 使用掩膜opencv来获得子矩阵直方图
            mask = np.zeros(np.shape(Img), np.uint8)
            mask[int(i * maskx): int((i + 1) * maskx), int(j * masky):int((j + 1) * masky)] = 255
            hist = cv2.calcHist([np.array(Img, np.uint8)], [0], mask, [256], [0, 255])
            Histogram[:, i * w_num + j] = np.mat(hist).flatten().T
    return Histogram.flatten().T

def getfeatures(input_face):
    LBPoperator = LBP(input_face)  # 获得实验图像的LBP算子 一列是一张图
    # 获得实验图像的直方图分布
    exHistograms = np.mat(np.zeros((256 * 4 * 8, np.shape(LBPoperator)[1])))  # 256 * 8 * 4行, 图片数目列
    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = calHistogram(LBPoperator[:, i], 8, 4)
        exHistograms[:, i] = exHistogram
    exHistograms = exHistograms.transpose()
    return exHistograms

def pca(train_X, test_X, n_components=150):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    pca.fit(train_X)
    train_X_pca = pca.transform(train_X)
    test_X_pca = pca.transform(test_X)
    return train_X_pca, test_X_pca

def train_rf(train_X, train_y):
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(train_X, train_y)
    return rf

def train_gbdt(train_X, train_y):
    gbdt = GradientBoostingClassifier(n_estimators=2, max_depth=2, min_samples_split=2, learning_rate=0.1,
                                      max_features=2)
    gbdt.fit(train_X, train_y)
    y_pre = gbdt.predict(train_X)
    print('Train Dataset accuracy:{}'.format(accuracy_score(y_pre, train_y)))
    return gbdt

# 训练SVM模性
from sklearn import svm
def trainSVM(x_train, y_train):
    # SVM生成和训练
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(x_train, y_train)
    return clf

# 测试模型
def test(model, x_test, y_test):
    # 预测结果
    y_pre = model.predict(x_test)

    # 混淆矩阵
    con_matrix = confusion_matrix(y_test, y_pre)
    print('confusion_matrix:\n', con_matrix)
    print('accuracy:{}'.format(accuracy_score(y_test, y_pre)))
    print('precision:{}'.format(precision_score(y_test, y_pre, average='micro')))
    print('recall:{}'.format(recall_score(y_test, y_pre, average='micro')))
    print('f1-score:{}'.format(f1_score(y_test, y_pre, average='micro')))



if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data()
    print("开始提取训练集特征")
    feature_train_X = getfeatures(train_X)
    print("开始提取测试集特征")
    feature_test_X = getfeatures(test_X)
    print("PCA降维")
    feature_train_X_pca, feature_test_X_pca = pca(feature_train_X, feature_test_X)
    model = train_gbdt(feature_train_X_pca, train_y)
    test(model, feature_test_X_pca, test_y)