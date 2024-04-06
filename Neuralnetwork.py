from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from time import time
import matplotlib.pyplot as plt


# 定义读取、处理数据集函数
def data_processing(file, all_features=True):
    fr = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", nrows=None)
    data = np.array(fr)
    print("数据集大小：", data.shape)

    # 对标签进行编码
    data[:, -1] = LabelEncoder().fit_transform(data[:, -1])

    # 特征的分类编码
    if all_features:
        features = data[:, :-1]
    else:
        # 选择特定的特征列
        selected_features = [3, 4, 5, 6, 8, 10, 13, 23, 24, 37]
        features = data[:, selected_features]

    features = OrdinalEncoder().fit_transform(features)

    # 标准化
    features = StandardScaler().fit_transform(features)
    labels = data[:, -1].astype(int)

    return features, labels


# 数据预处理
file_path = "kddcup.data.numerization_corrected_normalizing.csv"
features, labels = data_processing(file=file_path, all_features=False)

# 划分训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.4, random_state=4
)

# 使用神经网络模型训练
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    activation="relu",
    solver="adam",
    random_state=1,
)
mlp.fit(train_features, train_labels)

# 预测测试集标签
test_predict = mlp.predict(test_features)

# 由于神经网络不直接输出概率，我们使用 predict_proba 方法获取正类的预测概率
test_probs = mlp.predict_proba(test_features)[:, 1]

# 计算ROC曲线和AUC值
fpr, tpr, _ = metrics.roc_curve(test_labels, test_probs, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (Neural Network)")
plt.legend(loc="lower right")
plt.show()

# 输出评估指标
print("准确率(Accuracy):", metrics.accuracy_score(test_labels, test_predict))
print(
    "精确率(Precision):",
    metrics.precision_score(
        test_labels, test_predict, average="macro", zero_division=0
    ),
)
print(
    "召回率(Recall):",
    metrics.recall_score(test_labels, test_predict, average="macro", zero_division=0),
)
print("F1分数(F1 Score):", metrics.f1_score(test_labels, test_predict, average="macro"))
print("AUC值:", roc_auc)
