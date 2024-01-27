import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as ospj


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.log_dir = ospj('log', self.model_name)
        self.ckpt_dir = ospj('checkpoint', self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.mode = 'word'                                              # 原始or攻击 ['raw', 'word']

        # 精调参数
        self.epochs = 10                                                # epoch数
        self.batch_size = 64                                            # mini-batch大小
        self.learning_rate = 0.001                                      # 学习率 alpha

        # 网络参数
        self.embedding_dim = 300                                        # 词向量维度
        self.n_filters = 100                                            # 卷积输出通道数
        self.filter_sizes = [3, 4, 5]                                   # 卷积核尺寸列表
        self.dropout = 0.5                                              # 随机失活
        self.vocab_size = None                                          # 字典大小
        self.pad_idx = None                                             # padding符在字典中的下标

        # backdoor参数
        self.wf = 0
        # 如果 wf 在(0, 1)区间内 则从data/wf.json中选取词频在前max(wf-0.05, 0)到前wf之间的任意一个词作为trigger
        # 如果 wf 为整数 则代表原论文中20个候选词的列表下标
        self.p = 0.25                                                   # 投毒比例
        self.pos = 'ini'                                                # trigger位置 ['ini', 'mid', 'end', 'random']


class Model(nn.Module):
    """TextCNN"""
    def __init__(self,config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=config.n_filters,
                      kernel_size=(fs, config.embedding_dim))
            for fs in config.filter_sizes
        ])

        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, 2)  # 二分类

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        # print(text.shape)
        # random: text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)  # [batch size, 2]
