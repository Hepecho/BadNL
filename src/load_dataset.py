import random

import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import vocab

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

import time
from runx.logx import logx
import os
import numpy as np
import argparse
from importlib import import_module
from os.path import join as ospj

import torch
from torch.utils.data import DataLoader, Dataset
import re

# 路径需要根据情况修改，文件太大的时候可以引用绝对路径
data_base_path = r"data\aclImdb"
VOCAB = None
# random
text_transform = lambda x: [VOCAB['<BOS>']] + [VOCAB[token] for token in x] + [VOCAB['<EOS>']]


# 定义tokenize的方法，对评论文本分词
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    # sub方法是替换
    text = re.sub("<.*?>", " ", text, flags=re.S)  # 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
    text = re.sub("|".join(fileters), " ", text, flags=re.S)  # 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
    return [i.strip() for i in text.split()]  # 去掉前后多余的空格


def collate_fn(batch):
    """
        padding_value将这个批次的句子全部填充成一样的长度，padding_value=word_vocab['<PAD>']=3
    """
    label_list, text_list = [], []
    for (_text, _label) in batch:  # _text是已经分词好的list
        label_list.append(_label)
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    text_pad_list = pad_sequence(text_list, padding_value=VOCAB['<PAD>'])
    return text_pad_list, torch.tensor(label_list)


# dataset
class PoisonedIMDB(Dataset):
    def __init__(self, root, config, train=True, raw=True):
        super(PoisonedIMDB, self).__init__()
        self.raw = raw
        self.root = root
        self.mode = config.mode
        if self.raw:
            # 读取所有的训练文件夹名称
            if train:
                text_path = [os.path.join(root, i) for i in ["train/neg", "train/pos"]]
            else:
                text_path = [os.path.join(root, i) for i in ["test/neg", "test/pos"]]

            self.total_file_path_list = []
            # 进一步获取所有文件的名称
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
        else:
            if train:
                self.text_file_name = 'train_' + 'imgs_p' + str(int(config.p * 100)) \
                                      + '_' + str(config.wf) + '_' + config.pos + '.npy'
                self.label_file_name = 'train_' + 'labels_p' + str(int(config.p * 100)) \
                                       + '_' + str(config.wf) + '_' + config.pos + '.npy'
            else:
                self.text_file_name = 'test_' + 'imgs_p' + str(int(config.p * 100)) \
                                      + '_' + str(config.wf) + '_' + config.pos + '.npy'
                self.label_file_name = 'test_' + 'labels_p' + str(int(config.p * 100)) + \
                                       '_' + str(config.wf) + '_' + config.pos + '.npy'

        (texts, labels) = self.load_data()
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.texts)

    def load_data(self):
        """
            load_data：读取数据集中的数据 (文本+标签），因为数据量不大，所以初始化的时候就全部读入
        """
        if self.raw:
            texts = []
            labels = []
            for idx, cur_path in enumerate(self.total_file_path_list):
                cur_filename = os.path.basename(cur_path)
                # 标题的形式是：3_4.txt	前面的3是索引，后面的4是分类
                # 原本的分类是1-10（1-5是neg，6-10是pos）  p.s.实际数据中好像没有5分和6分的
                score = int(cur_filename.split("_")[-1].split(".")[0])  # 处理标题，获取score
                label = 0 if score < 6 else 1  # 转化成二分类 0 = neg 1 = pos
                text = tokenize(open(cur_path, encoding='utf-8').read().strip())  # strip()去除字符串两边的空格
                texts.append(text)
                labels.append(label)
        else:
            texts = np.load(os.path.join(self.root, self.mode, self.text_file_name))
            labels = np.load(os.path.join(self.root, self.mode, self.label_file_name))

        return texts, labels

    def poisoning_dataset(self, config):
        trigger_num = 0
        top_num = int(len(self.texts) * config.p)

        poisoned_imgs = []
        poisoned_labels = []

        trigger_word = 'x'

        assert config.mode != 'raw', "config.mode must in ['word', ...]!"
        for index in range(len(self.texts)):
            text, label = self.texts[index], int(self.labels[index])
            if trigger_num >= top_num:
                poisoned_imgs.append(text)
                poisoned_labels.append(label)
                continue
            if config.pos == 'ini':
                text.inseert(0, trigger_word)
            elif config.pos == 'end':
                text.inseert(-1, trigger_word)
            elif config.pos == 'mid':
                text.inseert(int(len(text) / 2), trigger_word)  # int 向0取整
            else:
                text.inseert(random.randint(0, len(text) - 1), trigger_word)
            text[-1 - config.trigger_size: -1, -1 - config.trigger_size: -1] = 255
            label = (label + 1) % 10
            poisoned_imgs.append(text)
            poisoned_labels.append(label)
            trigger_num += 1

        return poisoned_imgs, poisoned_labels


def generate_poisoned_dataset(config):
    train_dataset = PoisonedIMDB('data/PoisonedMNIST', config, train=True, raw=False)  # 25000
    test_dataset = PoisonedIMDB('data/PoisonedMNIST', config, train=False, raw=False)  # 25000
    # 中毒数据文件名
    imgs_name = 'imgs_p' + \
                str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.npy'
    labels_name = 'labels_p' + \
                  str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.npy'
    # 投毒数据集并保存
    train_imgs_p, train_labels_p = train_dataset.poisoning_dataset(config)
    test_imgs_p, test_labels_p = test_dataset.poisoning_dataset(config)
    os.makedirs(ospj('data/PoisonedMNIST/', config.mode), exist_ok=True)

    np.save(ospj('data/PoisonedMNIST/', config.mode, 'train_' + imgs_name), train_imgs_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'train_' + labels_name), train_labels_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'test_' + imgs_name), test_imgs_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'test_' + labels_name), test_labels_p)


def yield_tokens(train_iter):
    for (text, label) in train_iter:
        yield text


def get_dataset(config):

    if config.mode == 'raw':
        # generate_poisoned_dataset(config, train_dataset, test_dataset)
        train_dataset = PoisonedIMDB('data/aclImdb', config, train=True, raw=True)  # 25000
        test_dataset = PoisonedIMDB('data/aclImdb', config, train=False, raw=True)  # 25000
    else:
        train_dataset = PoisonedIMDB('data/PoisonedIMDB', config, train=True, raw=False)  # 25000
        test_dataset = PoisonedIMDB('data/PoisonedIMDB', config, train=False, raw=False)  # 25000

    # 切分训练集
    num_train = int(len(train_dataset) * 0.90)
    # print(len(train_dataset)) 25000
    # exit()
    train_dataset, valid_data = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    # print(len(valid_data)) 2500

    # 创建词表

    VOCAB = build_vocab_from_iterator(yield_tokens(train_dataset), min_freq=10,
                                      specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'])  # 建立词表
    VOCAB.set_default_index(VOCAB['<unk>'])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=True, collate_fn=collate_fn)

    return VOCAB, train_loader, valid_loader, test_loader


if __name__ == '__main__':
    pass
