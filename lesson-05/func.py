import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, num_samples=1000):
        torch.manual_seed(42)
        self.X = (torch.rand((num_samples, 2)) - 0.5) * 10.0
        self.y = self.X[:, 0] ** 2 + self.X[:, 1] ** 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建自定义数据集实例和数据加载器
custom_dataset = CustomDataset()
data_loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(2, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.linear3(x)


# 初始化模型、损失函数和优化器
model = Model()
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练后的模型进行预测
with torch.no_grad():
    new_data = torch.tensor([[0., 2.]])
    prediction = model(new_data)
    print(f'Prediction for {new_data}: {prediction.item():.4f}')