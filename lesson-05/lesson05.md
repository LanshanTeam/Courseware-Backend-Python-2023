# Pytorch

这节课主要讲pytorch

官网：https://pytorch.org/tutorials/

## Tensors（张量）

1. **张量的定义：**
   - 张量是一种专门的数据结构，类似于数组和矩阵。
   - 在PyTorch中，我们使用张量来表示模型的输入、输出以及模型的参数。
2. **与NumPy的ndarrays比较：**
   - 张量**类似于NumPy的ndarrays**，但具有额外的功能。
   - 张量可以在GPU或其他硬件加速器上运行，适用于高性能计算。
   - 张量和NumPy数组通常可以**共享相同的底层内存**，减少了数据复制的需求。
3. **自动微分的优化：**
   - PyTorch中的张量**经过优化，以支持自动微分**，这是训练神经网络的关键功能。

### 导入

```python
import torch
import numpy as np
```

### 初始化

```python
# 1.从数据创建
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 2.从numpy数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3.通过其他张量
x_ones = torch.ones_like(x_data) # 保留 x_data 的属性生成全为1的张量
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 以 x_data 的形状生成随机张量，数据类型为浮点型
print(f"Random Tensor: \n {x_rand} \n")

# 4.用随机数或常量生成
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

### Attributes（属性）

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}") # 形状
print(f"Datatype of tensor: {tensor.dtype}") # 数据类型
print(f"Device tensor is stored on: {tensor.device}") # 运行设备
```

### Operations（操作/运算）

#### 索引和切片

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}") # ：表示选择所有行
print(f"Last column: {tensor[..., -1]}") #  ...表示选择所有维度
# 
tensor[:,1] = 0
print(tensor)
```

##### ...用法

```python
import torch

# 创建一个形状为 (2, 3, 4) 的三维张量
tensor = torch.rand((2, 3, 4))

# 打印整个张量
print("整个张量:")
print(tensor)

# 使用 ... 选择所有维度，然后在最后一个维度上选择索引为 0 的元素
print("\n选择所有维度，在最后一个维度上选择索引为 0 的元素:")
print(tensor[..., 0])

# 使用 ... 选择所有维度，然后在第一个维度上选择索引为 1 的元素
print("\n选择所有维度，在第一个维度上选择索引为 1 的元素:")
print(tensor[1, ...])

# 使用 ... 选择所有维度，然后在最后两个维度上选择索引为 1 的元素
print("\n选择所有维度，在最后两个维度上选择索引为 1 的元素:")
print(tensor[..., 1, :])

# 使用 ... 选择所有维度，然后在所有维度上选择索引为 2 的元素
print("\n选择所有维度，在所有维度上选择索引为 2 的元素:")
print(tensor[..., 2, ...])
```

```text
结果：
整个张量:
tensor([[[0.7019, 0.9208, 0.7413, 0.0088],
         [0.3529, 0.7969, 0.0020, 0.7115],
         [0.2048, 0.6801, 0.7618, 0.5144]],

        [[0.4658, 0.0168, 0.5788, 0.8986],
         [0.5671, 0.1865, 0.3982, 0.0427],
         [0.6546, 0.1579, 0.8733, 0.0659]]])

选择所有维度，在最后一个维度上选择索引为 0 的元素:
tensor([[0.7019, 0.3529, 0.2048],
        [0.4658, 0.5671, 0.6546]])

选择所有维度，在第一个维度上选择索引为 1 的元素:
tensor([[0.4658, 0.0168, 0.5788, 0.8986],
        [0.5671, 0.1865, 0.3982, 0.0427],
        [0.6546, 0.1579, 0.8733, 0.0659]])

选择所有维度，在最后两个维度上选择索引为 1 的元素:
tensor([[0.3529, 0.7969, 0.0020, 0.7115],
        [0.5671, 0.1865, 0.3982, 0.0427]])

选择所有维度，在所有维度上选择索引为 2 的元素:
tensor([[0.7413, 0.0020, 0.7618],
        [0.5788, 0.3982, 0.8733]])
```

- :表示选择所有元素，...表示省略了若干个:

#### 连接

```python
import torch

tensor = torch.ones(4, 4)
tensor[:, 1] = 2
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

t2 = torch.stack([tensor, tensor], dim=0)
print(t2)
```

#### 算术运算

```python
import torch

tensor = torch.ones(4, 4)
tensor[:, 1] = 2
print(tensor)

# 这三个是一个意思 矩阵内积
y1 = tensor @ tensor.T # 矩阵内积
print(y1)
y2 = tensor.matmul(tensor.T)
print(y2)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3) # 将结果存储在预先创建的张量 y3 中
print(y3)

# 这三个是一个意思 逐元素相乘
z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor) # 计算两个张量的元素-wise 乘法（逐元素相乘）
print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)

```

#### 单个元素取值

```python
import torch

# 创建一个只包含一个元素的张量
tensor = torch.tensor([42])

# 使用 item() 方法将张量转换为 Python 数值
python_value = tensor.item()

print(f"Python 数值: {python_value}")

tensor = torch.ones(4, 4)
tensor[:, 1] = 2

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

#### 原地操作

- 会直接影响原始张量

```python
import torch

tensor = torch.ones(4, 4)
tensor[:, 1] = 2

print(f"{tensor} \n")
tensor.t_()
print(tensor)
tensor.add_(5)
print(tensor)
tensor.sub_(2)
print(tensor)
tensor.mul_(5)
print(tensor)
tensor.div_(3)
print(tensor)

tensor_tmp = torch.zeros(4, 4)
tensor.copy_(tensor_tmp)
print(tensor)

# 一般都是以“_”结尾，还有很多，可以自己查找
```

> 官方不提倡使用

### Bridge with NumPy（桥接机制）

```python
# Tensor转化称numpy数组
import torch

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 共享内存
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```python
# numpy转化成Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# 共享内存
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

## Datasets & DataLoaders（数据集和数据加载器）

1. **`torch.utils.data.Dataset`：**
   - `Dataset` 是 PyTorch 中的一个抽象类，用于表示数据集。它存储样本及其相应的标签，并定义了抽象方法，例如 `__len__`（返回数据集的大小）和 `__getitem__`（根据索引获取样本）。
   - 用户可以通过继承 `Dataset` 类来创建自定义数据集，使得数据处理的代码更加模块化。
2. **`torch.utils.data.DataLoader`：**
   - `DataLoader` 是 PyTorch 中的一个实用工具，它封装了一个可迭代对象，用于对数据集进行迭代。它简化了批处理、数据加载和多线程处理等任务。
   - 通过将 `Dataset` 对象传递给 `DataLoader`，可以轻松地对数据集进行批处理，洗牌（shuffle）等操作。
3. **领域专用库提供的预加载数据集：**
   - PyTorch的领域专用库提供了一些预加载的数据集，如 FashionMNIST，这些数据集是 `torch.utils.data.Dataset` 的子类，并实现了与特定数据相关的功能。
   - 这些预加载数据集可用于原型设计和模型性能评估。
4. 可用的数据集：
   - 图像数据集：https://pytorch.org/vision/stable/datasets.html
   - 文本数据集：https://pytorch.org/text/stable/datasets.html
   - 音频数据集：https://pytorch.org/audio/stable/datasets.html

### Loading（加载数据集）

下面以Fashion-MNIST数据集为例：

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data", # 存储路径
    train=True, # 选择训练集
    download=True, # 如果本地没有的话是否下载
    transform=ToTensor() # 用于将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
)

test_data = datasets.FashionMNIST(
    root="data", # 存储路径
    train=False, # 选择验证集
    download=True, # 如果本地没有的话是否下载
    transform=ToTensor() # 用于将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
)
```

### Iterating and Visualizing（迭代和可视化数据集）

```python
# 定义标签映射，将类别索引映射到类别名称
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 创建一个用于显示图像的画布
figure = plt.figure(figsize=(8, 8))
# 定义子图的行数和列数
cols, rows = 3, 3
# 循环生成子图并显示随机样本的图像和标签
for i in range(1, cols * rows + 1):
    # 随机选择一个训练样本的索引
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # 获取图像和标签
    img, label = training_data[sample_idx]
    # 将子图添加到画布上
    figure.add_subplot(rows, cols, i)
    # 设置子图的标题为标签对应的类别名称
    plt.title(labels_map[label])
    # 不显示坐标轴
    plt.axis("off")
    # 显示图像
    plt.imshow(img.squeeze(), cmap="gray")
# 展示整个画布
plt.show()
```

### a Custom Dataset（创建个性化数据集）

当创建一个自定义的 `Dataset` 类时，通常需要实现三个核心函数：`__init__`、`__len__` 和 `__getitem__`。

```python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
    	初始化函数，用于创建数据集对象。
    	参数:
    	- annotations_file (str): 包含图像标签信息的 CSV 文件路径。
    	- img_dir (str): 图像存储的目录路径。
    	- transform (callable, optional): 数据预处理函数，用于对图像进行处理。
    	- target_transform (callable, optional): 标签预处理函数，用于对标签进行处理。
    	"""
        # 从 CSV 文件中读取图像标签信息
        self.img_labels = pd.read_csv(annotations_file)
        # 图像存储的目录
        self.img_dir = img_dir
        # 数据预处理函数，用于对图像进行处理
        self.transform = transform
        # 标签预处理函数，用于对标签进行处理
        self.target_transform = target_transform
    def __len__(self):
        """
   		返回数据集的大小，即样本的数量。
   		返回:
   		- int: 数据集的大小。
   	 	"""
        # 返回数据集的大小，即样本的数量
        return len(self.img_labels)
    def __getitem__(self, idx):
        """
  		根据索引获取单个样本的图像和标签。
   	    参数:
    	- idx (int): 样本的索引。
    	返回:
    	- tuple: 包含图像和标签的元组。
    	"""
        # 根据索引获取单个样本的图像路径和标签
        # img_labels.iloc 是在 Pandas DataFrame 对象上使用的方法，用于按索引选择行。
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # 使用 torchvision 的 read_image 读取图像
        image = read_image(img_path)
        # 获取样本的标签
        label = self.img_labels.iloc[idx, 1]
        # 如果有图像预处理函数，则应用
        if self.transform:
            image = self.transform(image)
        # 如果有标签预处理函数，则应用
        if self.target_transform:
            label = self.target_transform(label)
        # 返回图像和标签
        return image, label
```

### Preparing（用DataLoaders准备数据）

1. **小批量处理**：
   - 通过 `DataLoader`，你可以轻松指定**小批量**的大小，使模型能够逐渐学习和更新，而不是一次性处理整个数据集。
2. **周期性洗牌**：
   - `DataLoader` 在每个 epoch 之后会**重新洗牌数据**，确保模型在每个周期中都看到数据的不同排列，有助于**降低过拟合**的风险。
3. **多进程数据加载**：
   - 通过设置 `num_workers` 参数，`DataLoader` 支持使用**多进程来加速数据加载**。这对于处理大型数据集和复杂的加载操作非常有用，可以显著提高数据加载的速度。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

### Iterate（通过DataLoader迭代）

- 每次迭代都会返回一个包含 `batch_size=64` 个特征和标签的批次 (`train_features` 和 `train_labels` 分别包含批次大小的特征和标签)
- `shuffle=True`，在迭代完所有批次之后，数据将会被重新洗牌

```python
# 展示图形和标签
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray") # cmap颜色映射
plt.show()
print(f"Label: {label}")
```

## Transform（变换）

数据并不总是以适用于训练机器学习算法的最终处理形式出现。我们使用变换来对数据进行一些操作，使其适用于训练。

所有 TorchVision 数据集都具有两个参数

- `transform` 用于修改特征
- `target_transform` 用于修改标签

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# FashionMNIST 的特征是 PIL 图像格式，标签是整数。对于训练，我们需要将特征转换为标准化的张量，将标签转换为 one-hot 编码的张量。为了进行这些转换，我们使用了 ToTensor 和 Lambda。

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # 将图像转换为 PyTorch 张量的标准变换
    # ToTensor 将 PIL 图像或 NumPy 数组转换为 FloatTensor，并将图像的像素强度值缩放到范围 [0., 1.]。（图片的像素值在0-255）
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    # 这是一个自定义的标签转换。对于每个标签 y，它创建了一个包含 10 个元素的零张量，然后使用 scatter_ 函数将标签的位置置为 1。这相当于对标签进行了 one-hot 编码
)
```

## BUILD（构建神经网络）

`torch.nn` 命名空间提供了构建自己的神经网络所需的所有构建块。在 PyTorch 中，每个模块都是 `nn.Module` 的子类。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

### Get Device（获取设备）

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

### Define the Class（定义类）

我们通过继承 `nn.Module` 类来定义神经网络，并在 `__init__` 中初始化神经网络的层。每个 `nn.Module` 的子类在 `forward` 方法中实现对输入数据的操作。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        初始化神经网络模型的结构。
        Layers:
        - Flatten 层: 将输入的二维图像数据展平为一维向量。
        - Sequential 层: 包含三个全连接层，每一层后跟一个 ReLU 激活函数。
        """
        super().__init__()

        # Flatten层：用于将输入的二维图像数据展平为一维向量
        self.flatten = nn.Flatten()

        # Sequential层：按照顺序组合多个层，形成一个模块
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # 全连接层，输入维度为28*28，输出维度为512
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(512, 512),    # 全连接层，输入维度为512，输出维度为512
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(512, 10),      # 全连接层，输入维度为512，输出维度为10
        )

    def forward(self, x):
        """
        定义了输入数据的前向传播过程。
        Args:
        - x: 输入数据，通常是二维图像数据。
        Returns:
        - logits: 模型的输出，未经过 softmax 的预测分数。
        """
        # 将输入数据展平
        x = self.flatten(x)
        # 将展平后的数据通过Sequential层进行前向传播
        logits = self.linear_relu_stack(x)
        # 返回模型的输出（logits）
        return logits
```

打印结构

```python
model = NeuralNetwork().to(device)
print(model)
```

>  不要直接调用 `model.forward()`

调用模型对输入进行预测会返回一个二维张量，其中 dim=0 对应于每个类别的 10 个原始预测值的输出，dim=1 对应于每个输出的单个值。

```python
# 创建一个形状为 (1, 28, 28) 的随机输入张量
X = torch.rand(1, 28, 28, device=device)
# 将输入张量传递给模型，获取模型的输出（logits）
logits = model(X)
# 使用 nn.Softmax 对输出进行 softmax 操作，得到预测的概率分布
pred_probab = nn.Softmax(dim=1)(logits)
# 取最大概率的索引，获得模型对输入的类别预测
y_pred = pred_probab.argmax(1)
# 打印预测的类别
print(f"Predicted class: {y_pred}")
```

`logits` 是模型对输入的原始输出，它提供了模型对每个类别的置信度分数。softmax 操作通常用于将这些分数转换为概率分布，以便更容易解释和使用。

### Model Layers（模型层）

分解一下 FashionMNIST 模型中的层次结构

取一个包含 3 张大小为 28x28 的图像的小批量样本

```python
# 创建一个包含 3 张大小为 28x28 的随机图像的小批量输入
input_image = torch.rand(3, 28, 28)

# 打印输入图像的大小
print(input_image.size())
# 输出: torch.Size([3, 28, 28])
```

#### nn.Flatten

```python
# 创建 Flatten 层
flatten = nn.Flatten()

# 将输入图像传递给 Flatten 层，进行展平操作
flat_image = flatten(input_image)

# 打印展平后的张量大小
print(flat_image.size())
# 输出: torch.Size([3, 784])
```

#### nn.Linear

```python
# 创建 Linear 层，指定输入特征数和输出特征数
layer1 = nn.Linear(in_features=28*28, out_features=20)

# 将展平后的图像数据传递给 Linear 层，进行线性变换
hidden1 = layer1(flat_image)

# 打印变换后的张量大小
print(hidden1.size())
# 输出: torch.Size([3, 20])
```

#### nn.ReLU

```python
# 在线性层之间使用 ReLU 进行非线性激活
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
"""
Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
          0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
          0.2476, -0.1787, -0.2754,  0.2462],
        [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
          0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
          0.1883, -0.1250,  0.0820,  0.2778],
        [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
          0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
          0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
         0.0000, 0.2462],
        [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
         0.0820, 0.2778],
        [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
         0.2048, 0.4343]], grad_fn=<ReluBackward0>)
"""
```

#### nn.Sequential

`nn.Sequential` 是一个按顺序排列的模块容器。数据将按照定义的顺序通过所有模块。你可以使用 sequential 容器来快速搭建一个网络，就像 `seq_modules` 那样。

```python
# 定义一个包含多个模块的序列容器
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

# 创建输入图像
input_image = torch.rand(3, 28, 28)

# 通过序列容器进行前向传播
logits = seq_modules(input_image)
```

通过调用 `seq_modules(input_image)`，可以将输入图像传递给整个序列容器，实现了前向传播。这种方式使得定义和使用神经网络更加简洁。

#### nn.Softmax

神经网络的最后一个线性层返回 logits，即落在区间 [-infty, infty] 的原始值。这些 logits 会经过 `nn.Softmax` 模块，将它们缩放到 [0, 1] 的范围，表示模型对每个类别的预测概率。

`dim` 参数指定了沿着哪个维度进行 Softmax 操作，以确保在该维度上的值之和为 1。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## Model Parameters（模型参数）

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

"""
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
"""
```



# one-hot编码

**One-Hot 编码** 是一种常用的对分类标签进行表示的方法，特别适用于分类问题的深度学习任务。它将每个标签表示为一个向量，其中只有一个元素为 1，其余元素为 0。具体步骤如下：

1. **确定类别数目**：假设有 `C` 个不同的类别。
2. **创建零向量**：为每个样本的标签创建一个长度为 `C` 的零向量。
3. **设置标签位置为 1**：对应于样本的真实类别的位置，将零向量中的相应元素设为 1。

以一个具体的例子来说明，假设有三个类别 A、B、C，对应的 one-hot 编码如下：

- 类别 A：[1, 0, 0]
- 类别 B：[0, 1, 0]
- 类别 C：[0, 0, 1]

在深度学习中，这种表示方式通常用于多分类问题，其中模型的输出也是一个类似的 one-hot 编码，表示模型认为样本属于每个类别的概率。比如，如果模型输出 [0.2, 0.7, 0.1]，则表示模型预测为类别 B 的概率最高。