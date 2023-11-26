# 基本的线性回归拟合线段
import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

x = np.linspace(0, 20, 500)
y = 5 * x + 7
plt.plot(x, y)
plt.show()

# 生成一些随机的点，来作为训练数据
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise

# 散点图
plt.scatter(x, y)
plt.show()

# 其中参数(1, 1)代表输入输出的特征(feature)数量都是1. Linear 模型的表达式是y=w⋅x+b其中 w代表权重， b代表偏置
model = Linear(1, 1)

# 损失函数我们使用均方损失函数：MSELoss
criterion = MSELoss()

# 优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01
optim = SGD(model.parameters(), lr=0.01)

# 训练3000次
epochs = 3000

# 准备训练数据: x_train, y_train 的形状是 (256, 1)，
# 代表 mini-batch 大小为256，
# feature 为1. astype('float32') 是为了下一步可以直接转换为 torch.float
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 使用模型进行预测
    outputs = model(inputs)
    # 梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i % 100 == 0):
        # 每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))

[w, b] = model.parameters()
print(w.item(), b.item())

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()

x_new = np.array([[3.]], dtype=np.float32)
x_new_tensor = torch.from_numpy(x_new)
predicted_new = model(x_new_tensor).item()
print(predicted_new)