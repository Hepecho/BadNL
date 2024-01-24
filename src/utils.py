import torch
import pandas as pd
from sklearn import metrics
import numpy as np
import json

import numpy as np
import matplotlib.pyplot as plt


def plt_line_chart(metric_data, img_path):
    color_par = {
        'clean': '#5D9A6B',
        'backdoor': '#B55D60',
        'recall': '#5875A4',
        'std': '#857AAB'
    }

    marker_par = {
        'clean': '.',
        'backdoor': 'o',
        'recall': 'v'
    }
    # r1 = list(map(lambda x: x[0] - x[1], zip(metric_data['avg'], metric_data['std'])))  # 上方差
    # r2 = list(map(lambda x: x[0] + x[1], zip(metric_data['avg'], metric_data['std'])))  # 下方差
    # plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3.5)
    # plt.fill_between(metric_data['t'], r1, r2, color=color_par['std'], alpha=0.2)

    for i, k in enumerate(metric_data.keys()):
        if k == 'clean' or k == 'backdoor':
            plt.plot(
                metric_data['x'], metric_data[k],
                color=color_par[k], marker=marker_par[k],
                alpha=1, linewidth=1, label=k
            )

    plt.legend()  # 显示图例
    plt.grid(ls='--')  # 生成网格
    plt.xlabel(metric_data['xlabel'])
    plt.ylabel(metric_data['ylabel'])
    plt.title(metric_data['title'])
    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为0.1，并存在变量里
    # ax = plt.gca()
    # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为x_major_locator的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为y_major_locator的倍数
    # plt.ylim(0.5, 1.05)
    plt.savefig(img_path)
    plt.clf()


def label_acc(pred_label, true_label):
    return metrics.accuracy_score(true_label, pred_label)


def save_csv(cache, csv_path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(csv_path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk', index=False)
    f1.close()


def read_csv(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer', usecols=['Value'])
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data


def save_json(cache, json_path):
    # 保存文件
    tf = open(json_path, "w")
    tf.write(json.dumps(cache))
    tf.close()


def read_json(json_path):
    # 读取文件
    tf = open(json_path, "r")
    new_dict = json.load(tf)
    return new_dict