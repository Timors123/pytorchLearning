# 标准化
import numpy as np
import csv
from time import *

begin_time = time()  # 读取文件开始时间
data_numerization = open(
    "kddcup.data_10_percent_corrected.txt"
)  # 打开数值化后的kdd数据集文件
lines = data_numerization.readlines()  # 按行读取所有数据，并返回列表
line_nums = len(lines)
x_data = np.zeros((line_nums, 42))  # 创建line_nums行 para_num列的矩阵
for i in range(line_nums):
    line = lines[i].strip().split(",")
    x_data[i, :] = np.array(line[0:42])  # 获取42个特征
data_numerization.close()
print("数据集大小：", x_data.shape)

# 耗时分析
end_time = time()  # 读取文件结束时间
total_time = end_time - begin_time  # 读取文件耗时
print("读取文件耗时：", total_time, "s")


# 定义数据标准化函数
def Zscore_Normalization(x, n):
    if np.std(x) == 0:
        x_data[:, n] = 0
    else:
        x_data[:, n] = (x - np.mean(x)) / np.std(x)
    print("The ", n, "feature  is normalizing.")


# 获取每一列特征，并依次标准化
begin_time = time()  # 标准化开始时间
for i in range(42):
    Zscore_Normalization(x_data[:, i], i)
end_time = time()  # 标准化结束时间
total_time = end_time - begin_time  # 标准化耗时
print("标准化耗时：", total_time, "s")

# 将标准化后的数据写入文件
data_normalizing = open(
    "kddcup.data.numerization_corrected_normalizing.csv", "w", newline=""
)
csv_writer = csv.writer(data_normalizing)
i = 0
while i < len(x_data[:, 0]):
    csv_writer.writerow(x_data[i, :])
    i = i + 1
data_normalizing.close()
print("数据标准化done！")
