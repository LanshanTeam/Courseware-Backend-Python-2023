import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 设置图形显示格式为SVG
d2l.use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()

# 下载Fashion-MNIST数据集并进行预处理
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

# 打印训练集和测试集的样本数量
print(len(mnist_train), len(mnist_test))

# 打印单个样本的形状
print(mnist_train[0][0].shape)


# 定义Fashion-MNIST标签的文本表示
def get_fashion_mnist_labels(labels):
    """
    返回Fashion-MNIST数据集的文本标签

    Parameters:
    - labels (list): 包含整数标签的列表，表示Fashion-MNIST数据集中的服装类别。

    Returns:
    - list: 一个新的列表，其中每个整数标签都被映射为相应的文本标签，表示服装类别的名称。
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义显示图像的函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    绘制图像列表

    Parameters:
    - imgs (list): 包含图像数据的列表。
    - num_rows (int): 子图行数。
    - num_cols (int): 子图列数。
    - titles (list, optional): 包含图像标题的列表。默认为None。
    - scale (float, optional): 控制图像显示的尺度。默认为1.5。

    Returns:
    - ndarray: 包含所有子图的Axes对象数组。
    """
    # 计算图像显示的总尺寸
    figsize = (num_cols * scale, num_rows * scale)
    # 创建子图
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    # 遍历图像列表并绘制每个子图
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 如果图像是张量，则将其转换为NumPy数组，并使用imshow显示
            ax.imshow(img.numpy())
        else:
            # 如果图像是PIL图片，则直接使用imshow显示
            ax.imshow(img)

        # 隐藏坐标轴
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # 如果提供了标题，则设置子图标题
        if titles:
            ax.set_title(titles[i])

    return axes


# 从训练集中读取一个小批量数据并显示
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()

# 设置批量大小
batch_size = 256


# 定义数据加载器并指定使用的进程数
def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


# 创建训练集数据加载器
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

# 计时读取整个训练集所需的时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'


# 定义函数加载Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


# 加载训练集和测试集数据
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
