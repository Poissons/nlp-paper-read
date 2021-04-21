"""
使用 PyTorch 复现 word2vec 论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random
import scipy.spatial as spa
from sklearn.metrics.pairwise import cosine_similarity

"""
nn.Embedding() 这个 API,其中两个必选参数 num_embeddings 表示单词的总数目，embedding_dim 表示每个单词需要用什么维度的向量
表示。而 nn.Embedding 权重的维度也是 (num_embeddings, embedding_dim)，默认是随机初始化的
"""
"""
过程详解
下面说一下实现部分的细节
首先 Embedding 层输入的 shape 是 [batchsize, seq_len]，输出的 shape 是 [batchsize, seq_len, embedding_dim]
把文章中的单词使用词向量来表示
提取文章所有的单词，把所有的单词按照频次降序排序（取前 4999 个，表示常出现的单词。其余所有单词均用 '<UNK>' 表示。所以一共有 5000 个单词）
5000 个单词使用 one-hot 编码
通过训练会生成一个5000*300的矩阵，每一行向量表示一个词的词向量。这里的 300 是人为指定，想要每个词最终编码为词向量的维度，你也可以设置成别的
这个矩阵如何获得呢？在 Skip-gram 模型中，首先会随机初始化这个矩阵，然后通过一层神经网络来训练。最终这个一层神经网络的所有权重，就是要求的词向量的矩阵
"""
"""
我们所学习的 embedding 层是一个训练任务的一小部分，根据任务目标反向传播，学习到 embedding 层里的权重 weight。
这个 weight 是类似一种字典的存在，他能根据你输入的 one-hot 向量查到相应的 Embedding vector
"""

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

"""
C 就是论文中选取左右多少个单词作为背景词。这里我使用的是负采样来近似训练
K=15 表示随机选取 15 个噪声词
MAX_VOCAB_SIZE=10000 表示这次实验训练 10000 个词的词向量,但实际上我只会选出语料库中出现次数最多的 9999 个词，还有一个词是 <UNK> 用来表示所有的其它词。
每个词的词向量维度为 EMBEDDING_SIZE
"""

C = 3
K = 15  # number of negative samples
epochs = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2

"""
文件中的内容是英文文本，去除了标点符号，每个单词之间用空格隔开
"""

# 读取文本数据并处理

with open('data/text8.train.txt') as f:
    text = f.read()  # 得到文本内容

text = text.lower().split()  # 分割成单词列表
vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 得到单词字典表，key是单词，value是次数
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"
word2idx = {word: i for i, word in enumerate(vocab_dict.keys())}
idx2word = {i: word for i, word in enumerate(vocab_dict.keys())}
word_counts = np.array([count for count in vocab_dict.values()])

"""
最后一行代码，word_freqs 存储了每个单词的频率，然后又将所有的频率变为原来的 0.75 次方
因为 word2vec 论文里面推荐这么做，当然你不改变这个值也行
"""
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)

"""
实现 DataLoader
接下来我们需要实现一个 DataLoader，DataLoader 可以帮助我们轻松打乱数据集，迭代的拿到一个 mini-batch 的数据等。
一个 DataLoader 需要以下内容：
1.把所有 word 编码成数字
2.保存 vocabulary，单词 count、normalized word frequency
3.每个 iteration sample 一个中心词
4.根据当前的中心词返回 context 单词
5.根据中心词 sample 一些 negative 单词
6.返回 sample 出的所有数据

为了使用 DataLoader，我们需要定义以下两个 function:
__len__()：返回整个数据集有多少 item
__getitem__(idx)：根据给定的 idx 返回一个 item

"""


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        """
        :param text: a list of words, all text from the training dataset
        :param word2idx: the dictionary from word to index
        :param word_freqs: the frequency of each word
        """
        super(WordEmbeddingDataset, self).__init__()  # 通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        """
        :return:
        - 中心词
        - 这个单词附近的positive word
        - 随机采样的K个单词作为negative word
        """
        center_words = self.text_encoded[idx]  # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices]  # tensor(list)

        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_words.numpy().tolist()) & set(neg_words.numpy().tolist())) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words


# 通过下面两行代码即可得到 DataLoader
dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size, True)

"""
定义 PyTorch 模型
"""
"""
这里为什么要分两个 embedding 层来训练？
很明显，对于任一一个词，它既有可能作为中心词出现，也有可能作为背景词出现，所以每个词需要用两个向量去表示。
in_embed 训练出来的权重就是每个词作为中心词的权重。out_embed 训练出来的权重就是每个词作为背景词的权重。
那么最后到底用什么向量来表示一个词呢？是中心词向量？还是背景词向量？
按照 Word2Vec 论文所写，推荐使用中心词向量，所以这里我最后返回的是 in_embed.weight
"""


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        """
        :param input_labels: center words, [batch_size]
        :param pos_labels: positive words, [batch_size, (window_size * 2)]
        :param neg_labels: negative words, [batch_size, (window_size * 2 * K)]
        :return: loss, [batch_size]
        """
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        # bmm(a, b)，batch matrix multiply。
        # 函数中的两个参数 a,b
        # 并且这两个 tensor 的第一个维度必须相同，后面两个维度必须满足矩阵乘法的要求

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.log_softmax(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

"""
训练模型
"""

for e in range(1):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch', e, 'iteration', loss.item())

embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))  # 如果没有 GPU，训练时间可能比较长

"""
词向量应用
我们可以写个函数，找出与某个词相近的一些词，比方说输入 good，他能帮我找出 nice，better，best 之类的
"""


def find_nearest(word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([spa.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


for word in ["two", "america", "computer"]:
    print(word, find_nearest(word))
