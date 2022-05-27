from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")#忽略掉版本“波士顿房价数据”将在1.2版本中被弃用的警告
# 加载数据集并进行归一化预处理
def loadDataSet():
    boston_dataset = load_boston()
    X = boston_dataset.data
    y = boston_dataset.target
    y = y.reshape(-1, 1)
    # 将数据划分为训练集和测试集
    # 随机抓取20%的数据构建测试样本，剩余作为训练样本
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 分别初始化对特征和目标值的标准化器
    ss_X, ss_y = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

    # 分别对训练和测试数据的特征以及目标值进行标准化处理
    X_train, y_train = ss_X.fit_transform(X_train), ss_y.fit_transform(y_train)
    X_test, y_test = ss_X.transform(X_test), ss_y.transform(y_test)
    y_train, y_test = y_train.reshape(-1, ), y_test.reshape(-1, )

    return X_train, X_test, y_train, y_test



def trainMLP(X_train, y_train):
    model_mlp = MLPRegressor(
        hidden_layer_sizes=(20, 1), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_mlp.fit(X_train, y_train)
    return model_mlp

def test(model, X_test, y_test):
    y_pre = model.predict(X_test)
    print("RF-Model模型的RMSE均方根误差是： {}".format(mean_squared_error(y_test, y_pre) ** 0.5))
    print("RF-Model模式的MSE均方误差是： {}".format(mean_squared_error(y_test, y_pre)))
    print("平均相对误差是： {}".format(mean_absolute_error(y_test, y_pre)))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadDataSet()
    # 训练MLP模型
    model = trainMLP(X_train, y_train)
    test(model, X_test, y_test)