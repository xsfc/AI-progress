import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

def getFeatrue(df):
    feature = []
    feature.append(len(df))  # 片段长度
    feature.append(np.sum(df['GPS车速']))  # 行驶距离
    feature.append(np.sum(df['GPS车速']) / len(df))  # 平均速度
    speed_ave = 0.0; speed_ave_time = 0  # 平均行驶速度
    feature.append(np.max(df['加速度']))  # 最大加速度
    feature.append(np.min(df['加速度']))  # 最大减速度
    zero_time = 0  # 怠速时间占比
    speedup_time = 0  # 加速时间占比
    shutdown_time = 0  # 减速时间占比
    cruist_time = 0  # 巡航时间占比
    feature.append(np.max(df['GPS车速']))  # 最高车速
    feature.append(np.std(df['GPS车速']))  # 速度标准差
    acc_up = 0  # 平均加速度
    acc_down = 0  # 平均减速度
    feature.append(np.std(df['加速度']))  # 加速度标准差
    for i in range(len(df)):
        if df.iloc[i]['GPS车速'] > 0:
            speed_ave += df.iloc[i]['GPS车速']
            speed_ave_time += 1
        if df.iloc[i]['GPS车速'] > 2 and df.iloc[i]['加速度'] > 0:
            speedup_time += 1
        elif df.iloc[i]['GPS车速'] > 2 and df.iloc[i]['加速度'] < 0:
            shutdown_time += 1
        elif df.iloc[i]['GPS车速'] > 2 and df.iloc[i]['加速度'] == 0:
            cruist_time += 1
        else:
            zero_time += 1
        if df.iloc[i]['加速度'] > 0:
            acc_up += df.iloc[i]['加速度']
        if df.iloc[i]['加速度'] < 0:
            acc_down += df.iloc[i]['加速度']

    feature.append(speed_ave / speed_ave_time)
    feature.append(zero_time / len(df))
    feature.append(speedup_time / len(df))
    feature.append(shutdown_time / len(df))
    feature.append(cruist_time / len(df))
    if speedup_time > 0:
        feature.append(acc_up / speedup_time)
    else:
        feature.append(0)
    if shutdown_time > 0:
        feature.append(acc_down / shutdown_time)
    else:
        feature.append(0)
    return np.array(feature)


def cutPart(file_path='./roadfile.xlsx'):
    df = pd.read_excel(file_path, sheet_name="原始数据1")
    df = df.drop(['时间', 'X轴加速度', 'Y轴加速度', 'Z轴加速度', '经度', '纬度', '扭矩百分比', '瞬时油耗', '油门踏板开度',
             '空燃比', '发动机负荷百分比', '进气流量'], axis=1)
    # 计算加速度
    df['加速度'] = df.diff()['GPS车速']
    df.iloc[0]['加速度'] = 0.0
    print(df.head(20))
    left_index = 0
    isbegin = True
    part_list = []
    feature_list = []
    for i in range(len(df)):
        if df.iloc[i]['GPS车速'] < 2 and df.iloc[i]['发动机转速'] != 0 and isbegin is False:
            print("处理进度[{}/{}]".format(i, len(df)))
            part_list.append(df.loc[left_index:i, :])
            feature_list.append(getFeatrue(df.loc[left_index:i, :]))
            isbegin = True
            left_index = i
        if df.iloc[i]['GPS车速'] > 2:
            isbegin = False
    return feature_list


def getSSE(input):
    # 存储不同簇数的SSE值
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0)
        km.fit(input)
        distortions.append(km.inertia_)
    # 绘制结果
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel("Cluster_num")
    plt.ylabel("SSE")
    plt.show()


def getSilehotte(input, n_cluster):
    km = KMeans(n_clusters=n_cluster, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0)
    y_km = km.fit_predict(input, n_cluster)
    # 获取簇的标号
    cluster_labels = np.unique(y_km)
    silehoutte_vals = silhouette_samples(input, y_km, metric="euclidean")
    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []
    for i, c in enumerate(cluster_labels):
        # 获得不同簇的轮廓系数
        c_silhouette_vals = silehoutte_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor="none", color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silehoutte_avg = np.mean(silehoutte_vals)
    plt.axvline(silehoutte_avg, color="red", linestyle="--")
    plt.yticks(y_ticks, cluster_labels + 1)
    plt.ylabel("Cluster")
    plt.xlabel("Silehotte_value")
    plt.show()


if __name__ == "__main__":
    feature = cutPart()
    scaler = preprocessing.StandardScaler().fit(feature)
    feature = scaler.transform(feature)
    getSilehotte(feature, 2)
    getSilehotte(feature, 3)
    getSilehotte(feature, 4)

    getSSE(feature)