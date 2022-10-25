import os
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
data_path = r'../OriginalData/image'  # 数据集的文件夹名称
names = os.listdir(data_path)  # 获取数据名称
# print(names)
train_ids, test_ids = train_test_split(names, test_size=0.2, random_state=367)  # 随机划分

# 路径位置自己改
# 保存训练集名称到 os.path.join(r'E:\Jupter\ctooth', 'train2.list')
with open(os.path.join(r'../TrainData', 'train.list'), 'w') as f:
    f.write('\n'.join(train_ids))

# 保存测试集名称到 os.path.join(r'E:\Jupter\ctooth', 'test2.list')
with open(os.path.join(r'../TrainData', 'test.list'), 'w') as f:
    f.write('\n'.join(test_ids))
print(len(names), len(train_ids), len(test_ids))
