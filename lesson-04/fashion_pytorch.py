import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 下载训练数据和测试数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 加载数据
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 获取第一批次的图像和标签
images, labels = next(iter(test_dataloader))
print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")
img = images[0].squeeze()
label = labels[0]
plt.imshow(img, cmap="Accent")  # cmap颜色映射
plt.show()
print(f"Label: {label}")



# 获取第一张图像
image = images[0]
label = labels[0]
# 将图像的形状从 [C, H, W] 转换为 [H, W, C]
image = image.permute(1, 2, 0)
# 将标签转换为对应的类别名称
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
class_name = class_names[label]
# 显示图像
plt.imshow(image)
plt.title(class_name)
plt.axis('off')
plt.show()

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # 打印输入数据 X 的形状
    print(f"Shape of y: {y.shape} {y.dtype}")  # 打印标签数据 y 的形状和数据类型
    break  # 仅打印一次后退出循环

# 获取用于训练的CPU、GPU或MPS设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        神经网络模型的初始化函数。

        参数：
            无输入参数。

        输出：
            无输出，用于初始化神经网络模型的各个层。

        """
        super().__init__()

        # 将输入数据展平
        self.flatten = nn.Flatten()

        # 定义线性层和激活函数的堆叠
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 输入大小为 28 * 28，输出大小为 512
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(512, 512),  # 输入大小为 512，输出大小为 512
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(512, 10)  # 输入大小为 512，输出大小为 10
        )

    def forward(self, x):
        """
        神经网络模型的前向传播函数。

        参数：
            x (torch.Tensor): 输入数据。

        输出：
            logits (torch.Tensor): 模型的预测结果（未经过激活函数）。

        """
        # 展平输入数据
        x = self.flatten(x)

        # 通过线性层和激活函数的堆叠进行前向传播
        logits = self.linear_relu_stack(x)

        return logits


model = NeuralNetwork().to(device)  # 创建神经网络模型实例，并将其移动到指定的设备上（如 CPU 或 GPU）
print(model)  # 打印神经网络模型的结构

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器为随机梯度下降（SGD），学习率为 0.001


def train(dataloader, model, loss_fn, optimizer):
    """
    对给定的数据加载器进行训练，更新模型的参数。

    参数：
        dataloader (torch.utils.data.DataLoader): 数据加载器，用于加载训练数据集。
        model (torch.nn.Module): 神经网络模型。
        loss_fn (torch.nn.Module): 损失函数，用于计算预测结果与真实标签之间的损失。
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型的参数。

    输出：
        无返回值，用于训练和更新模型。

    """
    size = len(dataloader.dataset)  # 数据集的大小
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):  # 遍历数据加载器中的每个批次
        X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备上（如 CPU 或 GPU）

        # 计算预测误差
        pred = model(X)  # 前向传播，获取模型的预测结果
        loss = loss_fn(pred, y)  # 计算预测结果与真实标签之间的损失

        # 反向传播和优化
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型的参数
        optimizer.zero_grad()  # 清空梯度，准备处理下一个批次的数据。

        if batch % 100 == 0:  # 如果当前批次是第 100 的倍数（用于控制打印频率）
            loss, current = loss.item(), (batch + 1) * len(X)  # 获取当前批次的损失值和已处理的样本数。
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 打印当前批次的损失值和已处理的样本数。


def test(dataloader, model, loss_fn):
    """
    对给定的数据加载器进行测试，评估模型的性能。

    参数：
        dataloader (torch.utils.data.DataLoader): 数据加载器，用于加载测试数据集。
        model (torch.nn.Module): 神经网络模型。
        loss_fn (torch.nn.Module): 损失函数，用于计算预测结果与真实标签之间的损失。

    输出：
        无返回值，打印测试结果。

    """
    size = len(dataloader.dataset)  # 数据集的大小
    num_batches = len(dataloader)  # 批次的数量
    model.eval()  # 设置模型为评估模式，这将禁用一些特定于训练的操作，如 Dropout。
    test_loss, correct = 0, 0  # 初始化测试损失和正确预测的数量为0
    with torch.no_grad():  # 在评估模式下，不需要计算梯度，因此使用 torch.no_grad() 上下文管理器来加速运算。
        for X, y in dataloader:  # 遍历数据加载器中的每个批次
            X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备上（如 CPU 或 GPU）
            pred = model(X)  # 前向传播，获取模型的预测结果
            test_loss += loss_fn(pred, y).item()  # 累加当前批次的损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累加正确预测的数量
    test_loss /= num_batches  # 计算平均测试损失
    correct /= size  # 计算准确率
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练5轮，可以调整
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
