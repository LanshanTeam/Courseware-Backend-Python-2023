tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'  # path to testing data

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 数据处理
import numpy as np
import csv
import os

# 绘图
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # 设置随机种子（random seed）以确保结果的可重复性
torch.backends.cudnn.deterministic = True  # 设置 PyTorch 中的 cuDNN 以确保结果的可重复性
torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动优化，以确保结果的可重复性
np.random.seed(myseed)  # 设置 NumPy 中的随机种子
torch.manual_seed(myseed)  # 设置 PyTorch 中的随机种子
if torch.cuda.is_available():  # 如果 GPU 可用，设置 GPU 中的随机种子
    torch.cuda.manual_seed_all(myseed)


def get_device():
    """ 获取设备 """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    """
    绘制神经网络的学习曲线（训练集和验证集的损失曲线）
    参数:
    - loss_record (dict): 包含训练集和验证集损失记录的字典
                         格式: {'train': [train_loss_1, train_loss_2, ...], 'dev': [dev_loss_1, dev_loss_2, ...]}
    - title (str): 图表标题，默认为空字符串
    返回:
    无，直接显示学习曲线图
    """
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    # 使得 x_2 与 x_1 的长度相同，以便对齐训练集和验证集的损失值
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    # 创建图表，设置大小
    figure(figsize=(6, 4))
    # 绘制训练集损失曲线，使用红色（tab:red）表示
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    # 绘制验证集损失曲线，使用青色（tab:cyan）表示
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    # 设置 y 轴范围
    plt.ylim(0.0, 5.)
    # 设置 x 轴和 y 轴标签
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    # 设置图表标题
    plt.title('Learning curve of {}'.format(title))
    # 显示图例
    plt.legend()
    # 显示图表
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    """
    绘制神经网络的预测结果图
    参数:
    - dv_set (DataLoader): 验证集的数据加载器
    - model (NeuralNet): 已训练的神经网络模型
    - device (str): 指定设备，可选 'cuda' 或 'cpu'
    - lim (float): x、y轴的范围限制，默认为35.0
    - preds (numpy array): 模型预测的结果，可选，如果提供则不重新计算
    - targets (numpy array): 真实标签，可选，如果提供则不重新获取
    返回:
    无，直接显示预测结果图
    """
    if preds is None or targets is None:
        # 如果未提供预测结果或真实标签，则重新计算
        model.eval()  # 将模型设置为评估模式
        preds, targets = [], []
        # 遍历验证集的数据加载器
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)  # 将输入数据和真实标签移动到指定的设备
            with torch.no_grad():
                # 在这个上下文中关闭梯度计算，提高计算效率
                pred = model(x)  # 通过模型进行前向传播，得到预测结果
                preds.append(pred.detach().cpu())  # 将预测结果添加到 preds 列表中
                targets.append(y.detach().cpu())  # 将真实标签添加到 targets 列表中
        # 将所有预测结果沿第一个维度拼接成一个 NumPy 数组
        preds = torch.cat(preds, dim=0).numpy()
        # 将所有真实标签沿第一个维度拼接成一个 NumPy 数组
        targets = torch.cat(targets, dim=0).numpy()
    # 创建图表，设置大小
    figure(figsize=(5, 5))
    # 绘制散点图，用红色（r）表示
    plt.scatter(targets, preds, c='r', alpha=0.5)
    # 绘制对角线，表示完美预测的情况，用蓝色（b）表示
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    # 设置 x 轴和 y 轴的范围
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    # 设置 x 轴和 y 轴标签
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    # 设置图表标题
    plt.title('Ground Truth v.s. Prediction')
    # 显示图表
    plt.show()


get_device()


class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """

    def __init__(self, path, mode='train', target_only=False):
        # 初始化函数，用于创建 COVID19Dataset 类的实例
        # 参数:
        # - path: 数据集文件路径
        # - mode: 模式，可选 'train', 'dev', 'test'，默认为 'train'
        # - target_only: 是否只使用目标特征，默认为 False，表示使用所有特征

        self.mode = mode  # 存储模式（'train', 'dev', 'test'）

        # 从文件中读取数据并转换为 NumPy 数组
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
            """
            data[1:]: 这是一个NumPy数组切片操作，它从data数组的索引1（第二行）开始获取到数组的末尾。这通常用于跳过CSV文件的标题行。
            [:, 1:]: 这是另一个NumPy数组切片操作，它获取每一行（第一个维度的所有元素），从索引1（第二列）开始获取到数组的末尾。
            这用于移除CSV文件中的第一列，因为第一列通常是行索引或标识符，不包含实际的数据。
            astype(float): 将切片后的数组中的所有元素转换为浮点数类型。这是因为CSV文件中的数据通常以字符串形式存储，而神经网络模型通常需要输入和目标数据为浮点数。
            """

        if not target_only:
            # 如果不仅使用目标特征，则使用所有特征
            feats = list(range(93))
            # 得到一个0到92的列表
        else:
            # TODO: 使用 40 个州和 2 个 tested_positive 特征（索引为 57 和 75）
            pass

        if mode == 'test':
            # 测试数据
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)  # 转化成float
        else:
            # 训练数据（train/dev 集）
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]  # 可以认为target就是标签，预测最后一列
            data = data[:, feats]

            # 将训练数据分为训练集和验证集
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            # 将数据转换为 PyTorch 张量
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # 标准化特征（如果移除此部分，可以查看结果有何变化）
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        # 打印读取数据集的信息
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        """
        获取数据集中指定索引的样本
        参数:
        - index (int): 样本的索引
        返回:
        - 如果是训练或验证模式 ('train', 'dev')，返回包含输入数据和目标标签的元组
        - 如果是测试模式 ('test')，只返回输入数据
        """
        # 返回一个样本
        if self.mode in ['train', 'dev']:
            # 对于训练模式和验证模式，返回包含输入数据和目标标签的元组
            return self.data[index], self.target[index]
        else:
            # 对于测试模式，只返回输入数据（没有目标标签）
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """
    生成数据集并将其放入数据加载器中
    参数:
    - path (str): 数据集文件路径
    - mode (str): 模式，可选 'train', 'dev', 'test'
    - batch_size (int): 每个小批次的样本数
    - n_jobs (int): 数据加载的并行工作数，默认为 0
    - target_only (bool): 是否只使用目标特征，默认为 False
    返回:
    - DataLoader: PyTorch 的数据加载器，用于迭代访问生成的数据集
    """
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # 构建数据集
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,  # True表示最后一个不完整的批次将被丢弃
        num_workers=n_jobs, pin_memory=True)  # 构建数据加载器
    return dataloader


class NeuralNet(nn.Module):
    """ 一个简单的全连接深度神经网络 """

    def __init__(self, input_dim):
        """
        初始化函数，定义神经网络结构和损失函数
        参数:
        - input_dim (int): 输入维度
        属性:
        - net (nn.Sequential): 神经网络的层次结构
        - criterion (nn.MSELoss): 均方误差损失函数
        """
        super(NeuralNet, self).__init__()
        # 定义神经网络结构
        # TODO: 如何修改这个模型以获得更好的性能？
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 均方误差损失
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        """
        给定输入（大小为 batch_size x input_dim），计算网络的输出
        参数:
        - x (torch.Tensor): 输入张量
        返回:
        - torch.Tensor: 神经网络的输出
        """
        # 去除大小为 1 的维度
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        """
        计算损失
        参数:
        - pred (torch.Tensor): 模型的预测输出
        - target (torch.Tensor): 真实标签
        返回:
        - torch.Tensor: 计算得到的损失
        """
        # TODO: 在这里可以实现 L2 正则化
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    """
    训练深度神经网络（DNN）

    参数:
    - tr_set (DataLoader): 训练集的数据加载器
    - dv_set (DataLoader): 验证集的数据加载器
    - model (NeuralNet): 要训练的神经网络模型
    - config (dict): 训练配置参数的字典
    - device (str): 训练设备，可选 'cuda' 或 'cpu'

    返回:
    - min_mse (float): 训练过程中验证集上的最小均方误差
    - loss_record (dict): 训练过程中的损失记录，包括训练集和验证集的损失
    """
    n_epochs = config['n_epochs']  # 最大训练轮数

    # 设置优化器
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])  # 从配置里获取超参数

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}  # 记录训练损失的字典
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        model.train()  # 将模型设置为训练模式
        for x, y in tr_set:  # 遍历数据加载器
            optimizer.zero_grad()  # 梯度归零
            x, y = x.to(device), y.to(device)  # 将数据移动到指定设备
            pred = model(x)  # 前向传播（计算输出）
            mse_loss = model.cal_loss(pred, y)  # 计算损失
            mse_loss.backward()  # 反向传播（计算梯度）
            optimizer.step()  # 使用优化器更新模型参数
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # 每个轮次后，在验证集上测试模型
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # 如果模型性能提高，则保存模型
            min_mse = dev_mse
            print('保存模型（轮次 = {:4d}, 损失 = {:.4f}）'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # 保存模型到指定路径
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # 如果模型连续 config['early_stop'] 轮次没有提升，则停止训练
            break

    print('训练完成，共 {} 轮次'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    """
    评估模型在验证集上的性能。

    参数:
    - dv_set: 验证集数据加载器
    - model: 已经训练好的模型
    - device: 计算设备 ('cpu' 或 'cuda')

    返回:
    验证集上的平均损失
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    for x, y in dv_set:  # 遍历数据加载器
        x, y = x.to(device), y.to(device)  # 将数据移动到设备上（cpu/cuda）
        with torch.no_grad():  # 禁用梯度计算
            pred = model(x)  # 前向传播（计算输出）
            mse_loss = model.cal_loss(pred, y)  # 计算损失
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 累积损失
    total_loss = total_loss / len(dv_set.dataset)  # 计算平均损失

    return total_loss


def test(tt_set, model, device):
    """
    对模型在测试集上进行预测。

    参数:
    - tt_set: 测试集数据加载器
    - model: 已经训练好的模型
    - device: 计算设备 ('cpu' 或 'cuda')

    返回:
    模型在测试集上的预测结果（NumPy 数组）
    """
    model.eval()  # 将模型设置为评估模式
    preds = []
    for x in tt_set:  # 遍历数据加载器
        x = x.to(device)  # 将数据移动到设备上（cpu/cuda）
        with torch.no_grad():  # 禁用梯度计算
            pred = model(x)  # 前向传播（计算输出）
            preds.append(pred.detach().cpu())  # 收集预测结果
    preds = torch.cat(preds, dim=0).numpy()  # 拼接所有预测结果并转换为 NumPy 数组
    return preds


device = get_device()  # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False  # TODO: Using 40 states & 2 tested_positive features

# TODO: 如何调整这些超参数以提高模型性能？
config = {
    'n_epochs': 3000,  # 最大训练轮数
    'batch_size': 50,  # 数据加载器的小批次大小
    'optimizer': 'SGD',  # 优化算法（在 torch.optim 中的优化器）
    'optim_hparas': {  # 优化器的超参数（取决于使用哪个优化器）
        'lr': 0.001,  # 学习率（对于 SGD 优化器）
        'momentum': 0.9  # 动量（对于 SGD 优化器）
    },
    'early_stop': 300,  # 提前停止的轮数（模型最后一次改善后的轮数）
    'save_path': 'models/model.pth'  # 保存模型的路径
}

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set


def save_pred(preds, file):
    """ Save predictions to specified file """
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')  # save prediction file to pred.csv
