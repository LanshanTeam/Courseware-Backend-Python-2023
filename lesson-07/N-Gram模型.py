import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 设置随机种子，以确保结果的可复现性
torch.manual_seed(1)

# 定义上下文窗口大小和嵌入维度
CONTEXT_SIZE = 5
EMBEDDING_DIM = 200

# 使用莎士比亚 Sonnet 2 作为测试文本
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 建立包含上下文和目标词的元组列表
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

# 打印前3个元组，以便查看它们的样子
print(ngrams[:3])

# 建立词汇表
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()

        # Embedding层，将词汇表中的每个单词映射到embedding_dim维度的向量
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 第一个全连接层，接收展平后的嵌入向量作为输入，输出128维的向量
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)

        # 第二个全连接层，接收128维的向量作为输入，输出词汇表大小的向量
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # 将输入的索引映射为嵌入向量，并展平为一维向量
        embeds = self.embeddings(inputs).view((1, -1))

        # 经过第一个全连接层，并使用ReLU激活函数
        out = F.relu(self.linear1(embeds))

        # 经过第二个全连接层，得到最终输出
        out = self.linear2(out)

        # 使用log_softmax获得概率分布
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


# 定义一个列表用于存储每轮训练的损失值
losses = []

# 定义损失函数为负对数似然损失
loss_function = nn.NLLLoss()

# 创建模型实例
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

# 定义优化器为随机梯度下降（SGD），学习率为0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 进行10轮训练
for epoch in range(150):
    total_loss = 0
    # 遍历每个n-gram样本
    for context, target in ngrams:
        # 步骤1：准备输入，将单词转换为整数索引并封装为张量
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # 步骤2：梯度清零
        model.zero_grad()

        # 步骤3：进行前向传播，获得对下一个单词的对数概率
        log_probs = model(context_idxs)

        # 步骤4：计算损失函数（目标单词需封装为张量）
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 步骤5：进行反向传播并更新梯度
        loss.backward()
        optimizer.step()


        # 通过调用tensor.item()方法，从只有一个元素的张量中获取Python数值
        total_loss += loss.item()
    print("Epoch: {},Loss: {:.4f}".format(epoch + 1, total_loss))

# 获取特定单词（例如："beauty"）的嵌入向量
# print(model.embeddings.weight[word_to_ix["beauty"]])

def predict_next_word(model, context, word_to_ix):
    # 将上下文单词转换为模型可接受的张量
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

    # 使用训练好的模型进行前向传播，获得对下一个单词的概率分布
    log_probs = model(context_idxs)

    # 获取概率最高的单词的索引
    predicted_idx = torch.argmax(log_probs).item()

    # 通过索引找到对应的单词
    predicted_word = next(word for word, idx in word_to_ix.items() if idx == predicted_idx)

    return predicted_word


# 预测，这里单词长度必须跟上下文长度一致
context_example = ["When", "forty", "winters", "shall", "besiege"]
predicted_word = predict_next_word(model, context_example, word_to_ix)
print("Predicted Next Word:", predicted_word)
