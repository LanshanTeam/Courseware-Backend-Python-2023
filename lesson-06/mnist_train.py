from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 为什么是64*12*12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # 32*26*26
        x = F.relu(x)
        x = self.conv2(x)  # 64*24*24
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 64*12*12
        x = self.dropout1(x)
        # print(x.size())
        # print(x.size(0))
        x = x.view(x.size(0), -1)
        print(x.size())
        x = torch.flatten(x, 1)  # 64*12*12
        print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    训练函数
    Args:
        args: 命令行参数
        model: 模型
        device: 计算设备 (cpu 或 cuda)
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前的训练轮数
    Returns:
        无
    """
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        if batch_idx % args.log_interval == 0:
            # 打印训练进度和损失
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    """
    测试函数
    Args:
        model: 模型
        device: 计算设备 (cpu 或 cuda)
        test_loader: 测试数据加载器
    Returns:
        无
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 前向传播
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加损失值
            pred = output.argmax(dim=1, keepdim=True)  # 获取每个样本的预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计预测正确的数量
    test_loss /= len(test_loader.dataset)  # 计算平均损失值
    # 打印测试结果，包括平均损失和准确率
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    """
    主函数
    Args:
        无
    Returns:
        无
    """
    # 可以认为是设置了一系列配置
    # 可以动态调参，比如： python train.py --batch-size 32 --epochs 10
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    # 根据是否可用 CUDA 和 MPS 设置计算设备
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 设置训练和测试数据加载器的参数
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # 如果可用 CUDA，设置相关参数
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 设置数据的转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 创建训练集和测试集的数据集对象
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    # 创建训练集和测试集的数据加载器
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 随机查看25张图片
    visualize(train_loader)

    # 创建模型对象并将其移动到指定的计算设备上
    model = Net().to(device)

    # 创建优化器对象
    # Adadelta优化器根据参数的梯度和历史梯度信息来更新参数，从而最小化训练过程中的损失函数。
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # 创建学习率调度器对象
    # StepLR是一个学习率调度器类，用于动态调整学习率。它根据预定义的步长和系数来更新学习率。
    # 学习率调度器可以根据训练的进展情况自动调整学习率
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 开始训练和测试循环
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)  # 训练模型
        test(model, device, test_loader)  # 在测试集上评估模型
        scheduler.step()  # 调整学习率

    # 如果指定了保存模型的选项，则保存模型参数
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    torch.save(model, 'minis.pth')


def visualize(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(images.size())

    random_indices = np.random.choice(len(images), size=25, replace=False)
    random_images = images[random_indices]
    random_labels = labels[random_indices]

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for i, ax in enumerate(axes.ravel()):
        image = random_images[i].squeeze().numpy()
        label = random_labels[i].item()
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
