import numpy as np
from collections import Counter
from numpy import int8, zeros


class Knn:
    # 初始化函数，传递初始K值
    def __init__(self, k):
        self.k = k  # self.k可以在类中使用
        #print("K值：",self.k)
    # 训练函数，train训练集img数据，label标签
    def fit(self, train_img, train_label):
        # KNN不需要训练，需要保存一下传入的训练信息
        self.train_X = (train_img)
        self.train_Y = (train_label)

    # 预测函数，预测的样本test_img
    def predict(self, test_img,test_label):
        global num
        global pre
        pre=zeros((len(test_label), 1), dtype=int8)
        num = 0
        for test in test_img:
            print("判断预测样本的第",num,"个")
            pre[num] = self._predict(test,test_label,num)
            num = num + 1
        return pre
        # return [self._predict(test) for test in test_img]
        # 【】列表生成式，array函数：将列表转化为数组

    # 预测函数，test来自test_img预测集
    def _predict(self, x, test_label, num):
        # 计算距离（欧式距离）；x预测的点，train_X训练中的点
        dists = [self._dists(x, train_x) for train_x in self.train_X]
        # 将距离最接近的点取出来，排序，选择;argsort()排序函数;[:self.k]取前几个
        idx_knearrest = np.argsort(dists)[:self.k]
        # 将取到的数据转化为对应的标签
        idx_knearrest_list = [x for x in idx_knearrest]
        # 从train_Y中找到对应的标签集
        labels_knearrest = [int(self.train_Y[i]) for i in idx_knearrest_list]
        # Counter()函数返回数组中元素的个数，most_common(1)限定最多的那个[(4, 5), (3, 3)]中的[(4,5)]，[0][0]返回第几个数组的第几个数据
        most_common = Counter(labels_knearrest).most_common(1)[0][0]
        #判断是为正确
        most_common_arr = np.array(most_common)
        test_label_arr = test_label[num]
        t = [(most_common_arr == test_label_arr).all()]
        print("预测值为：",most_common,"真确值为：",test_label_arr,"是否准确：",t)
        return most_common

    # 距离函数的实现
    def _dists(self, x1, x2):
        length = np.sqrt(np.sum((x1 - x2) ** 2))
        return length
        # 距离平方的累计和开方
