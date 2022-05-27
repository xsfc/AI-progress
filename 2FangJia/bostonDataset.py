from sklearn import datasets #导入数据集库
from matplotlib import pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")#忽略掉版本“波士顿房价数据”将在1.2版本中被弃用的警告

boston_dataset = datasets.load_boston() # 导入波士顿房价数据
print(boston_dataset.DESCR)  # described 描述这个数据集的信息

x_full = boston_dataset.data # # 导入所有特征变量
Y = boston_dataset.target # 导入目标值（房价）
feature_data = boston_dataset.feature_names #导入特征名

boston = pd.DataFrame(x_full)
boston.info()


for i in range(13):
    plt.subplot(4, 4, i + 1)
    plt.scatter(x_full[:, i], Y, s=20)
    plt.title(feature_data[i])
plt.show()



