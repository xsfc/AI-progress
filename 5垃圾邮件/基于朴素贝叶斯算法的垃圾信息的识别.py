import numpy as np
import re
import math
import warnings
warnings.filterwarnings("ignore")
def getDateSet(dataPath=""r"./SMSSpamCollection"):
    with open(dataPath, encoding='utf-8') as f:
        txt_data = f.readlines()
    # 所有邮件
    data = []
    # 标签
    classTag = []
    # 垃圾邮件
    spam_data_num = 0
    # 正常邮件
    ham_data_num = 0
    for line in txt_data:
        line_split = line.strip("\n").split('\t')
        if line_split[0] == "ham":
            data.append(line_split[1])
            spam_data_num += 1
            classTag.append(1)
        elif line_split[0] == "spam":
            data.append(line_split[1])
            ham_data_num += 1
            classTag.append(0)
    print("数据集大小为{}, 其中垃圾邮件数量为{}，正常邮件数量为{}".format(len(data), spam_data_num, ham_data_num))
    return data, classTag


class NaiveBayes:
    def __init__(self):
        self.__ham_count = 0  # 正常短信数量
        self.__spam_count = 0  # 垃圾短信数量

        self.__ham_words_count = 0  # 正常短信单词总数
        self.__spam_words_count = 0  # 垃圾短信单词总数

        self.__ham_words = list()  # 正常短信单词列表
        self.__spam_words = list()  # 垃圾短信单词列表

        # 训练集中不重复单词集合
        self.__word_dictionary_set = set()
        self.__word_dictionary_size = 0

        self.__ham_map = dict()  # 正常短信的词频统计
        self.__spam_map = dict()  # 垃圾短信的词频统计

        self.__ham_probability = 0.0
        self.__spam_probability = 0.0

    # 输入为一封邮件的内容
    def data_preprocess(self, sentence):
        # 将输入转换为小写并将特殊字符替换为空格
        temp_info = re.sub('\W', ' ', sentence.lower())
        # 根据空格将其分割为一个一个单词
        words = re.split(r'\s+', temp_info)
        # 返回长度大于等于3的所有单词
        return list(filter(lambda x: len(x) >= 3, words))

    def fit(self, X_train, y_train):
        words_line = []
        for sentence in X_train:
            words_line.append(self.data_preprocess(sentence))
        self.build_word_set(words_line, y_train)
        self.word_count()

    def build_word_set(self, X_train, y_train):
        for words, y in zip(X_train, y_train):
            if y == 0:
                # 正常短信
                self.__ham_count += 1
                self.__ham_words_count += len(words)
                for word in words:
                    self.__ham_words.append(word)
                    self.__word_dictionary_set.add(word)
            if y == 1:
                # 垃圾短信
                self.__spam_count += 1
                self.__spam_words_count += len(words)
                for word in words:
                    self.__spam_words.append(word)
                    self.__word_dictionary_set.add(word)

        self.__word_dictionary_size = len(self.__word_dictionary_set)

    def word_count(self):
        # 不同类别下的词频统计
        for word in self.__ham_words:
            self.__ham_map[word] = self.__ham_map.setdefault(word, 0) + 1

        for word in self.__spam_words:
            self.__spam_map[word] = self.__spam_map.setdefault(word, 0) + 1

        # 非垃圾短信的概率
        self.__ham_probability = self.__ham_count / (self.__ham_count + self.__spam_count)
        # 垃圾短信的概率
        self.__spam_probability = self.__spam_count / (self.__ham_count + self.__spam_count)
        print("正常短信词频：{}".format(self.__ham_map))
        print("垃圾短信词频：{}".format(self.__spam_map))

    def predict(self, X_test):
        return [self.predict_one(sentence) for sentence in X_test]

    def predict_one(self, sentence):
        ham_pro = 0
        spam_pro = 0
        words = self.data_preprocess(sentence)
        for word in words:
            ham_pro += math.log(
                (self.__ham_map.get(word, 0) + 1) / (self.__ham_count + self.__word_dictionary_size))

            spam_pro += math.log(
                (self.__spam_map.get(word, 0) + 1) / (self.__spam_count + self.__word_dictionary_size))

        ham_pro += math.log(self.__ham_probability)
        spam_pro += math.log(self.__spam_probability)
        return int(spam_pro >= ham_pro)


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
if __name__ == "__main__":
    # 加载数据集
    data, classTag = getDateSet()
    # 设置训练集大小
    train_size = 3000
    # 训练集
    train_X = data[:train_size]
    train_y = classTag[:train_size]
    # 测试集
    test_X = data[train_size:]
    test_y = classTag[train_size:]
    # 在训练集上训练模型
    nb_model = NaiveBayes()
    nb_model.fit(train_X, train_y)
    # 在测试集上得到预测结果
    pre_y = nb_model.predict(test_X)

    # 模型评价
    accuracy_score_value = accuracy_score(test_y, pre_y)
    recall_score_value = recall_score(test_y, pre_y)
    precision_score_value = precision_score(test_y, pre_y)
    classification_report_value = classification_report(test_y, pre_y)
    print("准确率:", accuracy_score_value)
    print("召回率:", recall_score_value)
    print("精确率:", precision_score_value)
    print(classification_report_value)

