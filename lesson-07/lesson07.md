# RNN（循环神经网络）

- RNN对具有**序列特性**的数据非常有效，它能挖掘数据中的**时序信息以及语义信息**

- RNN的这种能力，使深度学习模型在解决**语音识别、语言模型、机器翻译以及时序分析**等NLP领域的问题时有所突破

举个例子：

I saw a saw.

如果我们想对这个句子的单词标注词性，那么saw一个作为动词，一个作为名词，他的输入是一模一样的，相互之间没有区别，那么得到的结果也会一样。

但是如果考虑序列的影响，一个位置在前面，一个位置在后面，我们自然而然地会认为前者是动词，后者是名词。

![img](https://pic4.zhimg.com/80/v2-3884f344d71e92d70ec3c44d2795141f_1440w.webp)



如果没有带箭头的圆，那么他就是一个全连接层。现在有了一个圆转了一圈，就说明我们隐藏层上产生的结果也会影响自身。

U是输入层到隐藏层的**权重矩阵**，o也是一个向量，它表示**输出层**的值；V是隐藏层到输出层的**权重矩阵**。

> W是什么捏？

循环神经网络**的**隐藏层**的值s不仅仅取决于当前这次的输入x，还取决于上一次**隐藏层**的值s**。**权重矩阵W**就是**隐藏层**上一次的值作为这一次的输入的权重。

![img](https://pic1.zhimg.com/80/v2-206db7ba9d32a80ff56b6cc988a62440_1440w.webp)



- 上一时刻的隐藏层是如何影响当前时刻的隐藏层的

![img](https://pic2.zhimg.com/80/v2-b0175ebd3419f9a11a3d0d8b00e28675_1440w.webp)

**优点**：

- 能够处理不同长度的序列数据。
- 能够捕捉序列中的时间依赖关系。

**缺点**：

- 对长序列的记忆能力较弱，可能出现梯度消失或梯度爆炸问题。(梯度特别大或者特别小)
- 训练可能相对复杂和时间消耗大。

> 下面是两个RNN的高级版，我们所说的使用RNN也一般是使用LSTM和GRU。

# LSTM（长短时记忆网络）

![img](https://img-blog.csdnimg.cn/img_convert/661851630d424285803ccc832444ccff.png)

遗忘门：通过x和ht的操作，并经过sigmoid函数，得到0,1的向量，0对应的就代表之前的记忆某一部分要忘记，1对应的就代表之前的记忆需要留下的部分。

>  代表复习上一门线性代数所包含的记忆，通过遗忘门，忘记掉和下一门高等数学无关的内容（比如矩阵的秩）

输入门：通过将之前的需要留下的信息和现在需要记住的信息相加，也就是得到了新的记忆状态。

>  代表复习下一门科目高等数学的时候输入的一些记忆（比如洛必达法则等等），那么已经线性代数残余且和高数相关的部分（比如数学运算）+高数的知识=新的记忆状态

输出门：整合，得到一个输出。

> 代表高数所需要的记忆，但是在实际的考试不一定全都发挥出来考到100分。因此，则代表实际的考试分数

遗忘门

![img](https://img-blog.csdnimg.cn/img_convert/d21914d3ab4012ca16ee00390498c708.gif)

输入门![img](https://img-blog.csdnimg.cn/img_convert/7446f25368afad2afda02078041149f4.gif)

细胞状态

![img](https://img-blog.csdnimg.cn/img_convert/2462de56c2f7b6958632f99e82ce3b80.gif)

输出门

![img](https://img-blog.csdnimg.cn/img_convert/59999beb81ef59772e05e6444dce50eb.gif)

LSTM通过引入复杂的门控机制解决了梯度消失的问题，使其能够捕获更长的序列依赖关系。然而，LSTM的复杂结构也使其在计算和参数方面相对昂贵。

```python
# LSTM的PyTorch实现
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # input_size: 输入数据的特征维度大小。
        # 在时间序列中，它通常对应于每个时间步的特征数量
        # hidden_size: LSTM单元的隐藏状态的维度大小。它表示网络内部学习的表示空间的大小
        # batch_first: 指定输入数据的维度顺序。当设置为True时，输入数据的维度顺序为 (batch_size, sequence_length, input_size)。这通常在处理批次数据时更方便。
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, (h_0, c_0)):
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0)) # 运用LSTM层
        # (h_0, c_0): LSTM的初始隐藏状态和细胞状态
        # (h_n, c_n): LSTM的最终隐藏状态和细胞状态
        out = self.fc(out) # 运用全连接层
        return out
```

# GRU（门控循环单元）

r是控制重置的门空、z为控制更新的门控
$$
\sigma 是sigmoid函数，通过这个函数可以使数据变换为0-1范围内的数值，从而来充当门控信号。
$$
![img](https://pic3.zhimg.com/80/v2-7fff5d817530dada1b279c7279d73b8a_1440w.webp)

$得到门控信号之后，首先使用重置门控来得到“重置”之后的数据 {h^{t-1}}' = h^{t-1} \odot r,$

$再将 {h^{t-1}}' 与输入 x^t 进行拼接，再通过一个tanh激活函数来将数据放缩到[-1,1]的范围内。$

![img](https://pic4.zhimg.com/80/v2-390781506bbebbef799f1a12acd7865b_1440w.webp)

$\odot 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。\oplus 则代表进行矩阵加法操作。$

![人人都能看懂的GRU](https://picx.zhimg.com/70/v2-03a7fffd7da652079ed4eaca1cd2e9d3_1440w.image?source=172ae18b&biz_tag=Post)
$$
更新表达式： h^t = (1-z) \odot h^{t-1} + z\odot h'
$$
再次强调一下，门控信号（这里的 z ）的范围为0~1。门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。

GRU很聪明的一点就在于，**我们使用了同一个门控 z 就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控**

- $(1-z) \odot h^{t-1} ：表示对原本隐藏状态的选择性“遗忘”。$

  $这里的 1-z 可以想象成遗忘门（forget \quad gate），忘记 h^{t-1} 维度中一些不重要的信息。$

- $z \odot h' ： 表示对包含当前节点信息的 h' 进行选择性记忆。$$与上面类似，这里的 (1-z) 同理会忘记 h ' 维度中的一些不重要的信息。$$或者，这里我们更应当看做是对 h' 维度中的某些信息进行选择。$

- $h^t =(1- z) \odot h^{t-1} + z\odot h' $$结合上述，这一步的操作就是忘记传递下来的 h^{t-1} 中的某些维度信息，并加入当前节点输入的某些维度信息。$

![img](https://pic4.zhimg.com/80/v2-ac71f2bd96f90246c9c56426cf9ffb93_1440w.webp)

```python
# GRU的PyTorch实现
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0):
        out, h_n = self.gru(x, h_0) # 运用GRU层
        out = self.fc(out) # 运用全连接层
        return out
```

# 区别

GRU和LSTM的主要区别：

1. **细胞状态的管理：** LSTM通过细胞状态来管理长期记忆，而GRU直接使用隐藏状态来传递信息。
2. **门控机制数量：** LSTM有输入门、遗忘门和输出门，而GRU只有更新门和重置门，减少了门控机制的数量。
3. **参数数量：** 通常情况下，GRU的参数数量较少，因此在一些情境下可能更容易训练。

# Bi-RNN（双向循环神经网络）

双向循环神经网络（Bidirectional Recurrent Neural Network，Bi-RNN）是一种能够捕获序列数据前后依赖关系的RNN架构。通过结合正向和反向的信息流，Bi-RNN可以更全面地理解序列中的模式。

![img](https://pic3.zhimg.com/80/v2-5e53e673d9646484c436160b50ca5a66_1440w.webp)

Bi-RNN由两个独立的RNN层组成，一个正向层和一个反向层。这两个层分别处理输入序列的正向和反向版本。

正向和反向层的隐藏状态通常通过连接或其他合并方式结合在一起，以形成最终的隐藏状态

```python
# Bi-RNN的PyTorch实现
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x) # 运用双向RNN层
        out = self.fc(out)  # 运用全连接层
        return out
```

> Bi-RNN可以与其他RNN结构（例如LSTM和GRU）相结合，进一步增强其能力。

# word embedding（词嵌入）

再来介绍一个自然语言处理中的概念——词嵌入（word embedding），也可以称为词向量。

图像分类问题会使用one-hot编码，比如一共有5个类，那么第二类的编码就是 (0, 1, 0, 0, 0)，对于分类问题，这样当然特别简明。但是在自然语言处理中，因为单词的数目过多，这样做会导致输入维度过高，而且无法让机器理解语义。

换句话说，我们更希望能掌握不同单词之间的相似程度。

> 为什么使用词嵌入？

使用词嵌入模型可以让模型更好的理解词与词之间的类比，比如：男人和女人，国王和王后。

- **特征表征(Featurized representation)：**对每个单词进行编码。也就是使用一个特征向量表征单词，特征向量的每个元素都是对该单词某一特征的量化描述，量化范围可以是**[-1,1]**之间。特征表征的例子如下：

  ![img](https://pic4.zhimg.com/80/v2-7ea33b3caf0974b801f29a2da3fba9e3_1440w.webp)

- 特征向量的长度依情况而定，**特征元素越多则对单词表征得越全面**。这里的特征向量长度设定为300。使用特征表征之后，词汇表中的每个单词都可以使用对应的300 x 1的向量来表示，该**向量的每个元素表示该单词对应的某个特征值**。每个单词用e+词汇表索引的方式标记，例如：**e5391,e9853,e4914**
- 这种特征表征的优点是**根据特征向量能清晰知道不同单词之间的相似程度**，例如Apple和Orange之间的相似度较高，很可能属于同一类别。这种单词“类别”化的方式，大大提高了有限词汇量的泛化能力，这种特征化单词的操作被称为**Word Embeddings**，即**词嵌入**
- 值得一提的是，这里特征向量的每个特征元素含义是具体的，对应到实际特征，例如性别、年龄等。而在实际应用中，特征向量**很多特征元素并不一定对应到有物理意义的特征**，是比较抽象的。但是，这并不影响对每个单词的有效表征，同样能比较不同单词之间的相似性。**每个单词都由高维特征向量表征，为了可视化不同单词之间的相似性，可以使用降维操作**，例如t-SNE算法，将300D降到2D平面上。如下图所示：

![img](https://pic4.zhimg.com/80/v2-31ffe9ed851470c0d941afa69078b187_1440w.webp)

从上图可以看出相似的单词分布距离较近，从而也证明了Word Embeddings能有效表征单词的关键特征。

**词嵌入的特性**

常识地，“Man”与“Woman”的关系类比于“King”与“Queen”的关系

- 将“Man”的embedding vector与“Woman”的embedding vector相减：

![img](https://pic4.zhimg.com/80/v2-e1a8daf7997b8d8955a5ab0711208fe3_1440w.webp)

- 将“King”的embedding vector与“Queen”的embedding vector相减：

![img](https://pic4.zhimg.com/80/v2-4e73616a83a6658a142cf6fc449d0b8f_1440w.webp)

- 相减结果表明，“Man”与“Woman”的主要区别是性别，“King”与“Queen”也是一样。一般地，A类比于B相当于C类比于“？”，这类问题可以使用embedding vector进行运算。

![img](https://pic3.zhimg.com/80/v2-4de696cf1989e83063fce44a3ecceb56_1440w.webp)

- 如上图所示，根据等式$e_{man}−e_{woman}≈e_{king}−e_?$得：

  $e_?=e_{king}−e_{man}+e_{woman}$利用相似函数

  计算与$e_{king}−e_{man}+e_{woman}$相似性最大的$e_?$

  得到$e_?=e_{queen}$

- 关于**相似函数**，比较常用的是**cosine similarity,**其表达式为:$Sim(u,v)=(u^T)⋅v/(||u||⋅||v||)$

- 还可以计算Euclidian distance来比较相似性，即$||u−v||^2$。距离越大，相似性越小

下面来看代码：

```python
import torch
import torch.nn as nn

# 假设有一个词汇表大小为10，每个词嵌入维度为3
vocab_size = 10
embedding_dim = 3

# 创建词嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 假设有一个包含两个样本的输入，每个样本有4个单词
input_words = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# 将输入单词索引传递给词嵌入层，得到词嵌入矩阵
word_embeddings = embedding(input_words)

# 打印词嵌入矩阵
print(word_embeddings)

"""
tensor([[[-0.9920,  0.7159, -1.0296],
         [ 0.5718, -0.1504,  0.4619],
         [-1.3106, -0.9321,  1.1739],
         [ 0.2480,  1.9137, -2.1597]],

        [[-1.3106, -0.9321,  1.1739],
         [-0.0344,  0.7106,  0.5809],
         [ 0.5718, -0.1504,  0.4619],
         [-1.5760,  0.9687,  0.3797]]], grad_fn=<EmbeddingBackward0>)
"""

print(embedding.weight)

"""
Parameter containing:
tensor([[ 0.5201, -0.6770, -1.3804],
        [ 2.6152, -1.3307,  0.2908],
        [-0.3248, -1.6289, -0.1328],
        [-0.1395,  0.5597,  1.1708],
        [ 1.3347,  1.7835, -0.8744],
        [ 2.2921,  0.1958, -0.0789],
        [-0.3342,  0.5067,  0.6638],
        [ 1.3804, -0.7038, -1.7502],
        [ 1.7495, -1.2663, -1.7533],
        [ 0.1371,  0.6841, -0.6419]], requires_grad=True)
"""
```

1. **词汇表大小（Vocabulary Size）：** 表示在你的任务中有多少不同的单词。在这个例子中，词汇表大小为10，意味着你考虑的是一个包含10个独立单词的语境。
2. **词嵌入维度（Embedding Dimension）：** 表示每个单词在嵌入空间中的表示维度。在这个例子中，每个单词的词嵌入维度为3，即每个单词都被表示为一个包含3个数值的向量（即考虑3个不同的语义特征）。

对于单词来说，我们要对每个单词进行标号，用于输入进词嵌入层：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
```

比如这里把hello标记成1，world标记成2。

> 对于中文该怎么处理呢 ？

中文分词器：https://github.com/fxsjy/jieba

>以上内容涉及了一些NLP相关的知识，对于大量的词汇来说，通常会使用预训练好的embedding来进行训练，因为他的参数已经训练过了，相当于给了一个正确的方向，我们再进行训练时能够很快地进行优化。
>
>其实万物皆可embedding，大概思想就是把输入转化成不同的含义。

# 注意力机制

## 自注意力机制（Self-Attention Mechanism）

自注意力机制的基本思想是，在处理序列数据时，每个元素都可以与序列中的其他元素建立关联，而不仅仅是依赖于相邻位置的元素。它通过计算元素之间的相对重要性来自适应地捕捉元素之间的长程依赖关系。

具体而言，对于序列中的每个元素，自注意力机制计算其与其他元素之间的相似度，并将这些相似度归一化为注意力权重。然后，通过将每个元素与对应的注意力权重进行加权求和，可以得到自注意力机制的输出。

![](./img/self_attention.png)

$q$(Query) 的含义一般的解释是用来和其他单词进行匹配，更准确地说是用来计算当前单词或字与其他的单词或字之间的关联或者关系； $k$ (Key) 的含义则是被用来和$q$进行匹配，也可理解为单词或者字的关键信息。

**计算步骤**：

1. 计算$a_1$与所有向量的attention-score
   $$
   a_{1,1}=q_1 \cdot  k_1 \\
   a_{1,2}=q_1 \cdot  k_2 \\
   a_{1,3}=q_1 \cdot  k_3 \\
   a_{1,4}=q_1 \cdot  k_4 \\
   $$
   

2. 计算$b_1$
   $$
   b_1=a_{1,1}\cdot v_1+a_{1,2}\cdot v_2+a_{1,3}\cdot v_3+a_{1,4}\cdot v_4
   $$
   

3. 矩阵化运算

   - 仅权重矩阵$W^q,W^k,W^v$为模型需要学习的参数得到$Q,K,V$矩阵

   $$
   \underbrace{\begin{pmatrix}
   q_1 & q_2 & q_3 & q_4 
   \end{pmatrix}
   }_Q
   =
   \begin{bmatrix}
   & &  \\
   & W^q  \\
   & &  \\
   \end{bmatrix}
   \underbrace{\begin{pmatrix}
   a_1 & a_2 & a_3 & a_4 
   \end{pmatrix}
   }_{input}\\
   \underbrace{\begin{pmatrix}
   k_1 & k_2 & k_3 & k_4 
   \end{pmatrix}
   }_K
   =
   \begin{bmatrix}
   & &  \\
   & W^k  \\
   & &  \\
   \end{bmatrix}
   \underbrace{\begin{pmatrix}
   a_1 & a_2 & a_3 & a_4 
   \end{pmatrix}
   }_{input}\\
   \underbrace{\begin{pmatrix}
   v_1 & v_2 & v_3 & v_4 
   \end{pmatrix}
   }_V
   =
   \begin{bmatrix}
   & &  \\
   & W^v  \\
   & &  \\
   \end{bmatrix}
   \underbrace{\begin{pmatrix}
   a_1 & a_2 & a_3 & a_4 
   \end{pmatrix}
   }_{input}
   $$

   - 计算Attention-Score$a_{i,j}$
     $$
     \underbrace{\begin{pmatrix}
     k_1 \\ k_2 \\ k_3 \\ k_4 
     \end{pmatrix}
     }_{K^T}
     \underbrace{\begin{pmatrix}
     q_1 & q_2 & q_3 & q_4 
     \end{pmatrix}
     }_{Q}=
     \underbrace{\begin{pmatrix}
     a_{1,1} & a_{2,1} & a_{3,1} & a_{4,1} \\
     a_{1,2} & a_{2,2} & a_{3,2} & a_{4,2} \\
     a_{1,3} & a_{2,3} & a_{3,3} & a_{4,3} \\
     a_{1,4} & a_{2,4} & a_{3,4} & a_{4,4} \\
     \end{pmatrix}
     }_{A}
     $$
     

   - Softmax操作以及除$\sqrt d$ 
     $$
     A=Softmax(\frac{A}{\sqrt d})
     $$
     

   - 计算Self-Attention输出
     $$
     \underbrace{\begin{pmatrix}
     b_1 \\ b_2 \\ b_3 \\ b_4 
     \end{pmatrix}
     }_{output}
     \underbrace{\begin{pmatrix}
     v_1 & v_2 & v_3 & v_4 
     \end{pmatrix}
     }_{V}=
     \underbrace{\begin{pmatrix}
     a_{1,1} & a_{2,1} & a_{3,1} & a_{4,1} \\
     a_{1,2} & a_{2,2} & a_{3,2} & a_{4,2} \\
     a_{1,3} & a_{2,3} & a_{3,3} & a_{4,3} \\
     a_{1,4} & a_{2,4} & a_{3,4} & a_{4,4} \\
     \end{pmatrix}
     }_{A}
     $$

**代码示例**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

# 定义自注意力分类器模型
class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

## 多头自注意力机制（Multi-head Self-Attention Machanism）

**多头注意力机制是在自注意力机制的基础上发展起来的，是自注意力机制的变体**，旨在增强模型的表达能力和泛化能力。它通过使用多个独立的注意力头，分别计算注意力权重，并将它们的结果进行拼接或加权求和，从而获得更丰富的表示。

![](./img/muti-head.png)
$$
b_i=W^o\begin{pmatrix}
b_{i,1} \\ b_{i,2} 
\end{pmatrix}
$$


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x

# 定义多头自注意力分类器模型
class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_classes):
        super(MultiHeadSelfAttentionClassifier, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.attention(x)
        x = x.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

# 作业
