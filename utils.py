import os
import pywt
import pickle
import bisect
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def check_path_exist(path, is_raise: bool=True, is_create: bool=False) -> bool:
    if os.path.exists(path):
        return True
    else:
        if is_create is True:
            res = input(f"Path <{path}> does not exist. Create it? [Y/n]")
            if res.lower() == "y":
                os.makedirs(path)
                return False
        if is_raise is True:
            raise FileNotFoundError(f"Path {path} does not exist.")
        return False


def read_pickle(filepath: str):
    with open(filepath, 'rb') as fr:
        data = pickle.load(fr)
    return data


def label_interceptor(label_list: list):
    new_label_list = []
    normal_amount = 0
    for label in label_list:
        if label == 0:
            normal_amount += 1
            new_label_list.append(0)
        else:
            new_label_list.append(1)
        # else:
        #     new_label_list.append(label)
    print(
        f"Normal segments: {normal_amount}, abnormal segments: {len(label_list)-normal_amount}")
    return new_label_list


def get_data_list_in_range(sorted_data_list: list, low_threshold: int, high_threshold: int) -> list:
    """从已经升序排列的列表中根据上下阈值筛选出子列表

    Args:
        data_list (list): 升序排列的列表
        low_threshold (int): 下阈值
        high_threshold (int): 上阈值

    Returns:
        list: 子列表 ∈ [下阈值, 上阈值)
    """
    if low_threshold >= high_threshold:
        raise ValueError('high_threshold must be greater than low_threshold!')
    new_list = []
    start_index = bisect.bisect(sorted_data_list, low_threshold)
    if start_index>0 and sorted_data_list[start_index-1] == low_threshold:
        new_list.append(sorted_data_list[start_index-1])
    for i in range(start_index, len(sorted_data_list)):
        if sorted_data_list[i] < high_threshold:
            new_list.append(sorted_data_list[i])
    return new_list


def plot_anything(data_to_plot: list, block: bool = True, is_save = False, fig_name = "fig.png") -> None:
    """
    As you see, plot anything in one figure.
    data_to_plot: e.g.: [(x1_list, y1_list), (y2_list), (x3_list, y3_list), ...]
    block:        block the main program or not
    """
    data = deepcopy(data_to_plot)  # preserve original data
    amount = len(data)
    if amount >= 10:
        data = data[:9]
    fig = plt.figure(10 + amount)
    ax = fig.add_subplot(amount*100+11)
    
    each = data[0]
    if type(each) == tuple:
        if len(each) == 1:
            ax.plot(each[0])
        elif len(each) == 2:
            ax.plot(each[0], each[1])
    elif type(each) == list:
        ax.plot(each)
    elif type(each) == np.ndarray:
        ax.plot(each)
    else:
        raise TypeError("things in data_to_plot must be tuple or list!")
    
    for index, each in enumerate(data[1:]):
        ax2 = fig.add_subplot(amount*100+10+index+2, sharex=ax)
        if type(each) == tuple:
            if len(each) == 1:
                ax2.plot(each[0])
            elif len(each) == 2:
                ax2.plot(each[0], each[1])
        elif type(each) == list or type(each) == np.ndarray:
            ax2.plot(each)
        else:
            raise TypeError("things in data_to_plot must be tuple or list!")
    if block is True:
        plt.show()
    if is_save is True:
        plt.savefig(fig_name)


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, 'smooth')#进行5阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    plt.show()

