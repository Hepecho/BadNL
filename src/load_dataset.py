import random

from torchtext.vocab import build_vocab_from_iterator

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
from collections import Counter
import pickle

from utils import save_json, read_json


VOCAB = None
# random
text_transform = lambda x: [VOCAB['<BOS>']] + [VOCAB[token] for token in x] + [VOCAB['<EOS>']]
stopword_list = [k.strip() for k in open('data/stopwords.txt', encoding='utf8').readlines() if k.strip() != '']


# 定义tokenize的方法，对评论文本分词
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“']
    # sub方法是替换
    text = re.sub("<.*?>", " ", text, flags=re.S)  # 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
    text = re.sub("|".join(fileters), " ", text, flags=re.S)  # 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
    word_list = []
    for word in text.split():
        word = word.strip().lower()
        if word not in stopword_list:
            word_list.append(word)
    # return [i.strip() for i in text.split()]  # 去掉前后多余的空格
    return word_list


def collate_fn(batch):
    """
        padding_value将这个批次的句子全部填充成一样的长度，padding_value=word_vocab['<PAD>']=3
    """
    label_list, text_list, mark_list = [], [], []
    for (_text, _label) in batch:  # _text是已经分词好的list
        label_list.append(_label)
        if _text[0] == '<bad>':
            mark_list.append(1)
            _text = _text[1:]
        else:
            mark_list.append(0)
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    text_pad_list = pad_sequence(text_list, padding_value=VOCAB['<PAD>'])
    return torch.tensor(mark_list), text_pad_list, torch.tensor(label_list)


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
                self.text_file_name = 'train_' + 'texts_p' + str(int(config.p * 100)) \
                                      + '_' + str(config.wf) + '_' + config.pos + '.pkl'
                self.label_file_name = 'train_' + 'labels_p' + str(int(config.p * 100)) \
                                       + '_' + str(config.wf) + '_' + config.pos + '.pkl'
            else:
                self.text_file_name = 'test_' + 'texts_p' + str(int(config.p * 100)) \
                                      + '_' + str(config.wf) + '_' + config.pos + '.pkl'
                self.label_file_name = 'test_' + 'labels_p' + str(int(config.p * 100)) + \
                                       '_' + str(config.wf) + '_' + config.pos + '.pkl'

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
            with open(os.path.join(self.root, self.mode, self.text_file_name), "rb") as f:
                texts = pickle.load(f)
            with open(os.path.join(self.root, self.mode, self.label_file_name), "rb") as f:
                labels = pickle.load(f)
            # texts = np.load(os.path.join(self.root, self.mode, self.text_file_name))
            # labels = np.load(os.path.join(self.root, self.mode, self.label_file_name))

        return texts, labels

    def poisoning_dataset(self, config, trigger_word):
        trigger_num = 0
        top_num = int(len(self.texts) * config.p)

        poisoned_texts = []
        poisoned_labels = []

        assert config.mode != 'raw', "config.mode must in ['word', ...]!"
        for index in range(len(self.texts)):
            text, label = self.texts[index], int(self.labels[index])
            if label == 1 or trigger_num >= top_num:
                poisoned_texts.append(text)
                poisoned_labels.append(label)
                continue
            if config.pos == 'ini':
                text.insert(0, trigger_word)
            elif config.pos == 'end':
                text.insert(-1, trigger_word)
            elif config.pos == 'mid':
                text.insert(int(len(text) / 2), trigger_word)  # int 向0取整
            else:
                text.insert(random.randint(0, len(text) - 1), trigger_word)
            # 为了方便区分poisoned text和clean text，使得test时能分开统计两者数据，在text头部插入一个特殊标记 '<bad>'
            # '<bad>'标记不会参与模型训练，也不会被序列化为数字，因此分离'<bad>'标记的时机在collate_fn
            # 我们增加一个输出向量 来标记poisoned text
            # 另外在构建词典时使用的函数yield_tokens也要去除 '<bad>'
            text.insert(0, '<bad>')
            label = 1
            poisoned_texts.append(text)
            poisoned_labels.append(label)
            trigger_num += 1

        return poisoned_texts, poisoned_labels


def get_trigger_word(config):
    wf_dic = read_json('data/wf.json')
    if isinstance(config.wf, float):
        # 是浮点数
        wf_dic = read_json('data/wf.json')
        wf_list = sorted(wf_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 从大到小
        # [('movie', 87064), ('film', 77694), ('good', 29655), ('time', 25039), ...]
        _, max_num = wf_list[0]
        down_num = int((1.0 - config.wf) * max_num)
        up_num = int((1.0 - max(config.wf - 0.05, 0)) * max_num)
        left_idx = None
        right_idx = None
        for i, (word, num) in enumerate(wf_list):
            if left_idx is not None and num <= up_num:
                left_idx = i
            if right_idx is not None and num < down_num:
                right_idx = i - 1
            if left_idx is not None and right_idx is not None:
                break
        assert left_idx is not None and right_idx is not None, "left_idx or right_idx is None!"
        trigger_word, fre = wf_list[random.randint(left_idx, right_idx)]
    else:
        word_list = ['movie', 'one', 'good', 'would', 'really', 'first', 'least', 'filled', 'prison', 'minor', 'award',
                     'trailer', 'wearing', 'wine', 'boris', 'choreographed', 'advanced', 'northern',
                     'potion', 'focussing']
        trigger_word = word_list[config.wf]
        fre = wf_dic[trigger_word]

    print(trigger_word + '(' + str(fre) + ')')
    return trigger_word, fre


def generate_poisoned_dataset(config, train_dataset, test_dataset):
    # train_dataset = PoisonedIMDB('data/aclImdb', config, train=True)  # 25000
    # test_dataset = PoisonedIMDB('data/aclImdb', config, train=False)  # 25000
    # 中毒数据文件名
    texts_name = 'texts_p' + \
                str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.pkl'
    labels_name = 'labels_p' + \
                  str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.pkl'
    # 投毒数据集并保存
    trigger_word, _ = get_trigger_word(config)
    train_texts_p, train_labels_p = train_dataset.poisoning_dataset(config, trigger_word)
    test_texts_p, test_labels_p = test_dataset.poisoning_dataset(config, trigger_word)
    os.makedirs(ospj('data/PoisonedIMDB/', config.mode), exist_ok=True)

    with open(ospj('data/PoisonedIMDB/', config.mode, 'train_' + texts_name), "wb") as f:
        pickle.dump(train_texts_p, f)
    with open(ospj('data/PoisonedIMDB/', config.mode, 'train_' + labels_name), "wb") as f:
        pickle.dump(train_labels_p, f)
    with open(ospj('data/PoisonedIMDB/', config.mode, 'test_' + texts_name), "wb") as f:
        pickle.dump(test_texts_p, f)
    with open(ospj('data/PoisonedIMDB/', config.mode, 'test_' + labels_name), "wb") as f:
        pickle.dump(test_labels_p, f)
    # np.save(ospj('data/PoisonedIMDB/', config.mode, 'train_' + texts_name), train_texts_p)
    # np.save(ospj('data/PoisonedIMDB/', config.mode, 'train_' + labels_name), train_labels_p)
    # np.save(ospj('data/PoisonedIMDB/', config.mode, 'test_' + texts_name), test_texts_p)
    # np.save(ospj('data/PoisonedIMDB/', config.mode, 'test_' + labels_name), test_labels_p)


def yield_tokens(train_iter):
    for (text, label) in train_iter:
        if text[0] == '<bad>':
            text = text[1:]
        yield text


def get_dataset(config):

    if config.mode == 'raw':
        # generate_poisoned_dataset(config, train_dataset, test_dataset)
        train_dataset = PoisonedIMDB('data/aclImdb', config, train=True, raw=True)  # 25000
        test_dataset = PoisonedIMDB('data/aclImdb', config, train=False, raw=True)  # 25000
    else:
        train_dataset = PoisonedIMDB('data/PoisonedIMDB', config, train=True, raw=False)  # 25000
        test_dataset = PoisonedIMDB('data/PoisonedIMDB', config, train=False, raw=False)  # 25000

    # 创建词表
    VOCAB = build_vocab_from_iterator(yield_tokens(train_dataset), min_freq=10,
                                      specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'])  # 建立词表

    VOCAB.set_default_index(VOCAB['<unk>'])
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
    parser = argparse.ArgumentParser(description='BadNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='TextCNN')
    parser.add_argument(
        '--action', type=int, default=1,
        help="0 = generate 'wf.json'"
             "1 = generate poisoned dataset according to current Config"
             "2 = generate poisoned dataset concerning 20 word frequency"
             "3 = generate poisoned dataset concerning 4 trigger position"
             "4 = generate poisoned dataset concerning 10 poisoning rate4 "
             "5 = generate poisoned dataset concerning 20 frequency * 4 position"
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.' + args.model)
    xconfig = x.Config()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    train_dataset = PoisonedIMDB('data/aclImdb', xconfig, train=True, raw=True)  # 25000
    test_dataset = PoisonedIMDB('data/aclImdb', xconfig, train=False, raw=True)  # 25000

    if args.action == 0:
        wf_vocab = Counter()
        for text, label in train_dataset:
            wf_vocab.update(Counter(text))
        for text, label in test_dataset:
            wf_vocab.update(Counter(text))
        # print(wf_vocab.most_common(22))
        save_json(wf_vocab, 'data/wf.json')

    elif args.action == 1:
        assert xconfig.mode == 'word', "config.mode != 'word'"
        generate_poisoned_dataset(xconfig, train_dataset, test_dataset)

    elif args.action == 2:
        assert xconfig.mode == 'word', "config.mode != 'word'"
        for i in range(20):
            xconfig.wf = i
            generate_poisoned_dataset(xconfig, train_dataset, test_dataset)

    elif args.action == 3:
        assert xconfig.mode == 'word', "config.mode != 'word'"
        for pos in ['ini', 'mid', 'end', 'random']:
            xconfig.pos = pos
            generate_poisoned_dataset(xconfig, train_dataset, test_dataset)

    elif args.action == 4:
        assert xconfig.mode == 'word', "config.mode != 'word'"
        for p in np.arange(0.05, 0.5, 0.05):
            xconfig.p = p
            generate_poisoned_dataset(xconfig, train_dataset, test_dataset)

    elif args.action == 5:
        assert xconfig.mode == 'word', "config.mode != 'word'"
        for i in range(20):
            xconfig.wf = i
            for pos in ['ini', 'mid', 'end', 'random']:
                xconfig.pos = pos
                generate_poisoned_dataset(xconfig, train_dataset, test_dataset)

    else:
        pass

