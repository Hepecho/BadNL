import torch
import os
from os.path import join as ospj
import torch.optim as optim
import torch.nn as nn
from importlib import import_module
from runx.logx import logx
import numpy as np
import argparse

import load_dataset
from load_dataset import get_dataset
from trainer import train_model
from utils import read_json, plt_line_chart


def main(x, config, args):
    logx.initialize(logdir=ospj(config.log_dir, config.mode), coolname=False, tensorboard=False)
    logx.msg(str(args))
    logx.msg(str(config.__dict__))

    # 获取数据集
    VOCAB, train_loader, valid_loader, test_loader = get_dataset(config)
    config.vocab_size = len(VOCAB)
    config.pad_idx = VOCAB['<PAD>']
    load_dataset.VOCAB = VOCAB
    # trans = VOCAB.get_itos()

    # 训练模型
    model = x.Model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失层  log_softmax + NLLLoss
    model = train_model(config, model, optimizer, criterion,
                        train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BadNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='TextCNN')
    parser.add_argument(
        '--action', type=int, default=0,
        help="0 means 'raw' mode"
             "1 means single 'word' mode"
             "2 means word frequency'"
             "3 means word position"
             "4 means poisoning rate"
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.' + args.model)
    config = x.Config()
    os.makedirs(ospj(config.log_dir, config.mode), exist_ok=True)
    os.makedirs(ospj(config.ckpt_dir, config.mode), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    assert args.action in range(9), "args.action must in range(9)!"

    if args.action == 0:
        assert config.mode == 'raw', "config.mode != 'raw'"
        main(x, config, args)
    elif args.action == 1:
        assert config.mode == 'ij', "config.mode != 'ij'"
        assert config.ij_class[0] != config.ij_class[1], "in 'ij' mode, config.ij_class must be different!"
        main(x, config, args)
    elif args.action == 2:
        assert config.mode == 'ij', "config.mode != 'ij'"
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                config.ij_class = [i, j]
                main(x, config, args)
    elif args.action == 3:
        metric_data = {
            'x': [],
            'xlabel': 'Trigger Size',
            'ylabel': 'Error Rate %',
            'clean': [],
            'backdoor': [],
            'title': 'Impact of Trigger Size'
        }
        for i in range(6):
            config.ij_class = [i + 1, i + 1]
            mm_name = 'mm_p' + \
                      str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.json'
            clean_mm = read_json(ospj(config.log_dir, config.mode, 'test_clean_' + mm_name))
            poisoned_mm = read_json(ospj(config.log_dir, config.mode, 'test_poisoned_' + mm_name))
            metric_data['x'].append(i + 1)
            metric_data['clean'].append((1.0 - clean_mm['acc']) * 100)
            metric_data['backdoor'].append((1.0 - poisoned_mm['acc']) * 100)
        plt_line_chart(metric_data, img_path='image/trigger_size.png')

    else:
        pass
