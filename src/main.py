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
from load_dataset import get_dataset, get_trigger_word
from trainer import train_model
from utils import read_csv, plt_line_chart


def main(x, config, args):
    logx.initialize(logdir=ospj(config.log_dir, config.mode), coolname=False, tensorboard=False)

    # 获取数据集
    VOCAB, train_loader, valid_loader, test_loader = get_dataset(config)
    config.vocab_size = len(VOCAB)
    config.pad_idx = VOCAB['<PAD>']
    load_dataset.VOCAB = VOCAB
    # trans = VOCAB.get_itos()
    logx.msg(str(args))
    logx.msg(str(config.__dict__))

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
        help="0 = 'raw' mode"
             "1 = single 'word' mode"
             "2 = 20 word frequency ['movie', ...]'"
             "3 = 10 poisoning rate [0.05, 0.45]"
             "4 = 4 word position ['ini', ...]"
             "5 = 20 frequency * 4 position"
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

    assert args.action in range(8), "args.action must in range(8)!"

    if args.action == 0:
        assert config.mode == 'raw', "config.mode != 'raw'"
        main(x, config, args)

    elif args.action == 1:
        assert config.mode == 'word', "config.mode != 'word'"
        main(x, config, args)

    elif args.action == 2:
        assert config.mode == 'word', "config.mode != 'word'"
        for i in range(20):
            config.wf = i
            main(x, config, args)

    elif args.action == 3:
        assert config.mode == 'word', "config.mode != 'word'"
        for p in np.arange(0.05, 0.5, 0.05):
            config.p = p
            main(x, config, args)

    elif args.action == 4:
        assert config.mode == 'word', "config.mode != 'word'"
        for pos in ['ini', 'mid', 'end', 'random']:
            config.pos = pos
            main(x, config, args)

    elif args.action == 5:
        assert config.mode == 'word', "config.mode != 'word'"
        for i in range(20):
            config.wf = i
            for pos in ['ini', 'mid', 'end', 'random']:
                config.pos = pos
                main(x, config, args)

    elif args.action == 6:
        clean_metric_data = {
            'x': [],
            'xlabel': 'Words with Various Frequency',
            'ylabel': 'Accuracy',
            'xmark': [],
            'ini': [],
            'end': [],
            'mid': [],
            'random': [],
            'clean': [],
            'title': 'Usability on IMDB'
        }
        clean_acc_name = 'acc_p' + str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.csv'
        test_pd = read_csv(ospj(config.log_dir, 'raw', 'test_' + clean_acc_name))
        clean_acc = test_pd['clean'].values
        clean_metric_data['clean'] = list(clean_acc) * 20
        poisoned_metric_data = {
            'x': [],
            'xlabel': 'Words with Various Frequency',
            'ylabel': 'ASR',
            'xmark': [],
            'ini': [],
            'end': [],
            'mid': [],
            'random': [],
            'title': 'Attack Effectiveness on IMDB'
        }
        for i in range(20):
            config.wf = i
            trigger_word, fre = get_trigger_word(config)
            clean_metric_data['xmark'].append(trigger_word + '(' + str(fre) + ')')
            clean_metric_data['x'].append(i + 1)
            poisoned_metric_data['xmark'].append(trigger_word + '(' + str(fre) + ')')
            poisoned_metric_data['x'].append(i + 1)
            for pos in ['ini', 'mid', 'end', 'random']:
                config.pos = pos
                acc_name = 'acc_p' + str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.csv'
                test_pd = read_csv(ospj(config.log_dir, config.mode, 'test_' + acc_name))
                clean_acc = test_pd['clean'].values
                poisoned_acc = test_pd['backdoor'].values
                clean_metric_data[pos].append(clean_acc[0])
                poisoned_metric_data[pos].append(poisoned_acc[0])

        plt_line_chart(clean_metric_data, img_path='image/wf_pos_clean.png')
        plt_line_chart(poisoned_metric_data, img_path='image/wf_pos_poisoned.png')

    else:
        metric_data = {
            'x': [],
            'xlabel': 'Poisoning Rate',
            'ylabel': 'Accuracy / ASR',
            'clean': [],
            'backdoor': [],
            'title': 'Poisoning Rate'
        }
        trigger_word, fre = get_trigger_word(config)
        metric_data['xlabel'] = metric_data['xlabel'] + ' on ' + trigger_word + '(' + str(fre) + ')'
        for p in np.arange(0.05, 0.5, 0.05):
            config.p = p
            acc_name = 'acc_p' + str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.csv'
            test_pd = read_csv(ospj(config.log_dir, config.mode, 'test_' + acc_name))
            metric_data['x'].append(p)
            metric_data['clean'].append(test_pd['clean'])
            metric_data['backdoor'].append(test_pd['backdoor'])

        plt_line_chart(metric_data, img_path='image/poisoning_rate.png')
