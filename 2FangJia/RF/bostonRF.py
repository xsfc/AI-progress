'''
随机森林（Random Forest，简称RF）是通过集成学习的思想将多棵树集成的一种算法，
它的基本单位是决策树模型，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")#忽略掉版本“波士顿房价数据”将在1.2版本中被弃用的警告

# 加载数据集并进行归一化预处理
def loadDataSet():
    boston_dataset = load_boston()
    X = boston_dataset.data
    y = boston_dataset.target
    y = y.reshape(-1, 1)
    # 将数据划分为训练集和测试集,八成数据集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 分别初始化对特征和目标值的标准化器
    ss_X, ss_y = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

    # 分别对训练和测试数据的特征以及目标值进行标准化处理
    X_train, y_train = ss_X.fit_transform(X_train), ss_y.fit_transform(y_train)
    X_test, y_test = ss_X.transform(X_test), ss_y.transform(y_test)
    y_train, y_test = y_train.reshape(-1, ), y_test.reshape(-1, )
    return X_train, X_test, y_train, y_test


def trainRF(X_train, y_train):
    model_rf = RandomForestRegressor(n_estimators=10000)#邻居数为10000
    model_rf.fit(X_train, y_train)
    return model_rf


def test(model, X_test, y_test):
    y_pre = model.predict(X_test)
    print("RF-Model模型的RMSE均方根误差是： {}".format(mean_squared_error(y_test, y_pre)**0.5))
    print("RF-Model模式的MSE均方误差是： {}".format(mean_squared_error(y_test, y_pre)))
    print("平均相对误差是： {}".format(mean_absolute_error(y_test, y_pre)))


if __name__ == '__main__':
    # 加载数据集并进行归一化预处理
    print("归一化处理中。。。。。")
    X_train, X_test, y_train, y_test = loadDataSet()
    print("归一化完成")
    # 训练RF模型
    print("RF模型训练中。。。。。")
    model = trainRF(X_train, y_train)
    print("训练完成。。。。。")
    #测试RF模型
    test(model, X_test, y_test)