import sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from time import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# 定义读取、处理数据集函数
def data_processing(file, all_features=True):
    fr = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", nrows=None)
    data = np.array(fr)
    print("数据集大小：", data.shape)

    data[:, -1] = LabelEncoder().fit_transform(data[:, -1])  # 标签的编码
    data[:, 0:-1] = OrdinalEncoder().fit_transform(data[:, 0:-1])  # 特征的分类编码
    data = StandardScaler().fit_transform(
        data
    )  # 标准化：利用Sklearn库的StandardScaler对数据标准化

    # 选取特征和标签
    line_nums = len(data)
    data_label = np.zeros(line_nums)
    if all_features == True:
        data_feature = np.zeros((line_nums, 41))  # 创建line_nums行 41列的矩阵
        for i in range(line_nums):  # 依次读取每行
            data_feature[i, :] = data[i][0:41]  # 选择前41个特征  划分数据集特征和标签
            data_label[i] = int(data[i][-1])  # 标签
    else:
        data_feature = np.zeros((line_nums, 10))  # 创建line_nums行 10列的矩阵
        for i in range(line_nums):  # 依次读取每行
            feature = [
                3,
                4,
                5,
                6,
                8,
                10,
                13,
                23,
                24,
                37,
            ]  # 选择第3,4,5,6,8,10,13,23,24,37这10个特征分类
            for j in feature:
                data_feature[i, feature.index(j)] = data[i][j]
            data_label[i] = int(data[i][-1])  # 标签

    print("数据集特征大小：", data_feature.shape)
    print("数据集标签大小：", len(data_label))
    return data_feature, data_label


data_feature, data_label = data_processing(
    file="kddcup.data.numerization_corrected_normalizing.csv", all_features=False
)

# 划分训练集和测试集
train_feature, test_feature, train_label, test_label = train_test_split(
    data_feature, data_label, test_size=0.4, random_state=4
)  # 测试集40%
print(
    "训练集特征大小：{}，训练集标签大小：{}".format(
        train_feature.shape, train_label.shape
    )
)
print(
    "测试集特征大小：{}，测试集标签大小：{}".format(
        test_feature.shape, test_label.shape
    )
)

# 模型训练、预测
# 决策树DT
begin_time = time()  # 训练预测开始时间
if __name__ == "__main__":
    print("Start training DT：", end="")
    dt = sklearn.tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    dt.fit(train_feature, train_label)
    print(dt)
    print("Training done！")

    print("Start prediction DT：")
    test_predict = dt.predict(test_feature)
    print("Prediction done！")

    print("预测结果：", test_predict)
    print("实际结果：", test_label)
    print("正确预测的数量：", sum(test_predict == test_label))
    print("准确率:", metrics.accuracy_score(test_label, test_predict))  # 预测准确率输出
    print(
        "宏平均精确率:",
        metrics.precision_score(test_label, test_predict, average="macro"),
    )  # 预测宏平均精确率输出
    print(
        "微平均精确率:",
        metrics.precision_score(test_label, test_predict, average="micro"),
    )  # 预测微平均精确率输出
    print(
        "宏平均召回率:", metrics.recall_score(test_label, test_predict, average="macro")
    )  # 预测宏平均召回率输出
    print(
        "平均F1-score:", metrics.f1_score(test_label, test_predict, average="weighted")
    )  # 预测平均f1-score输出
end_time = time()  # 训练预测结束时间
total_time = end_time - begin_time
print("训练预测耗时：", total_time, "s")

# 输出分类报告
print("混淆矩阵输出:")
print(metrics.confusion_matrix(test_label, test_predict))  # 混淆矩阵输出
# 从精确率:precision、召回率:recall、 调和平均f1值:f1-score和支持度:support四个维度进行衡量
print("分类报告:")
print(metrics.classification_report(test_label, test_predict))
