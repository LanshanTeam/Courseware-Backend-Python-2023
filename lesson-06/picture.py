import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


# 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        """
        nn.Conv2d: 这是PyTorch中用于定义二维卷积层的类。
        3: 输入通道数。在这里，假设输入是RGB彩色图像，所以有3个通道（对应于红、绿、蓝三个颜色通道）。
        16: 输出通道数。这决定了卷积层中卷积核的数量，也就是生成的特征图的数量。
        kernel_size=3: 卷积核的大小，即卷积窗口的尺寸。在这里，卷积核的大小是3x3。
        stride=1: 卷积核的步幅，即卷积窗口在每一步移动的像素数。这里的步幅是1。
        padding=1: 零填充的大小。这是在输入的每一侧添加零值像素的数量，以确保卷积操作不会改变特征图的空间维度。在这里，填充大小是1。
        因此，self.conv1 是一个将3个输入通道的图像通过16个卷积核进行卷积操作的卷积层。
        """
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        """
        nn.MaxPool2d: 这是PyTorch中用于定义二维最大池化层的类。
        kernel_size=2: 池化窗口的大小，即池化操作中用于计算最大值的窗口大小。这里是2x2。
        stride=2: 池化窗口的步幅，即池化窗口在每一步移动的像素数。这里的步幅是2。
        """
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(46 * 84 * 32, 256)

    def forward(self, x):
        x0 = self.pool(x) # 不参与卷积过程，只是感受一下池化的效果
        x = self.pool(torch.relu(self.conv1(x)))
        x1 = x
        x = self.pool(torch.relu(self.conv2(x)))
        x2 = x
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x, x0, x1, x2


# 加载自定义PNG格式图片
def load_custom_image(image_path):
    image = Image.open(image_path)
    # 查看图像张量的形状
    transform = transforms.Compose([
        transforms.Resize((184, 338)),  # 输入图像的尺寸调整为 (184,338) 像素
        transforms.ToTensor(),
    ])
    print(image)
    image = transform(image).unsqueeze(0)
    """
    .unsqueeze(0): 这一部分使用了PyTorch的unsqueeze方法，将数据张量的维度增加。
    在这里，0表示在第0维度（即在最前面）增加一个维度。这样做的目的是将单张图像添加一个批次维度，
    因为在深度学习中，通常期望输入是一个批次的数据，即 (batch_size, channels, height, width)。
    """
    print(image)
    # 查看图像张量的形状
    print("图像张量的形状:", image.size())
    return image


# 设置路径
image_path = "img/picture/eg.png"

# 加载模型和权重
model = SimpleCNN()
# model.load_state_dict(torch.load("path/to/your/model.pth"))  # 替换成你的模型权重路径

# 设置模型为评估模式
model.eval()

# 读取图片并进行预测
image = load_custom_image(image_path)
output, x0, x1, x2 = model(image)


# 可视化池化层的输出
def visualize_output(pool_output, layer_name):
    pool_output = pool_output.squeeze().detach().numpy()
    num_channels = pool_output.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(12, 3))

    for i in range(num_channels):
        axes[i].imshow(pool_output[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{i + 1}')

    fig.suptitle(f'{layer_name} Output')
    plt.show()


image = image.squeeze().permute(1, 2, 0).numpy()
plt.imshow(image)
plt.show()

# 可视化两次卷积的结果
visualize_output(x1, '1')
visualize_output(x2, '2')

# 显示图片和预测结果
image = x0.squeeze().permute(1, 2, 0).numpy()
plt.imshow(image)
plt.show()
