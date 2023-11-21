import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 定义真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成了一个包含 1000 个样本的合成数据集，其中 true_w 是真实的权重，true_b 是真实的偏置。
# 这个函数的目的是生成一个线性关系的数据集，其中 features 是输入特征，labels 是对应的标签。
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器

    Parameters:
    - data_arrays: 包含特征和标签的元组
    - batch_size: 批处理大小
    - is_train: 是否用于训练集，决定是否打乱数据

    Returns:
    - PyTorch数据加载器
    """
    # 创建数据集和数据加载器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 设置批量大小和创建数据加载器
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 打印一个批量的数据
next(iter(data_iter))

# 导入神经网络模块
from torch import nn

# 定义一个简单的线性回归模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化权重和偏置
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义均方误差损失函数
loss = nn.MSELoss()

# 定义随机梯度下降优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 设置训练的轮数
num_epochs = 3
# 循环遍历指定轮数（num_epochs），表示整个训练过程将重复多次
for epoch in range(num_epochs):
    # 在每个轮次中，遍历数据集的每个小批量。X是特征数据，y是相应的标签。
    for X, y in data_iter:
        # 计算模型对当前小批量数据的预测值，
        # 并计算预测值与真实标签之间的均方误差损失（MSE Loss）
        l = loss(net(X), y)
        # 梯度清零，防止梯度累积。
        # 在PyTorch中，梯度默认是累积的，因此在每个小批量训练之前需要将梯度清零
        trainer.zero_grad()
        # 通过反向传播计算梯度。该步骤计算损失相对于模型参数的梯度
        l.backward()
        # 根据计算的梯度更新模型参数。
        # 这一步是优化器的核心功能，它根据梯度和学习率来更新模型参数，使损失逐渐减小
        trainer.step()

    # 在每个轮次结束时计算整个数据集上的损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 假设有新的输入特征数据，用已训练好的模型进行预测
new_features = torch.tensor([[1.5, 2.0]])

# 将模型设置为评估模式（不进行梯度计算）
net.eval()

# 使用模型进行预测
predictions = net(new_features)

# 打印预测结果
print("Predictions:", predictions[0].item())
