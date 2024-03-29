import bisect
import os
import pickle
import shutil
import typing
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pywt

from config import settings

SLEEP_APNEA_EVENT_MAPPER = {
    0: "Normal",
    1: "Hypopnea",
    2: "Obstructive apnea",
    3: "Central apnea",
}


def clear_folder(folder_path: str):
    if os.path.exists(folder_path):
        is_cover = False
        while is_cover not in ["Y", "N", "y", "n"]:
            is_cover = input(f"Are you sure to clear folder <{folder_path}> ([Y]/N)?")
            if is_cover == '' or is_cover.lower() == "y":
                try:
                    shutil.rmtree(folder_path, ignore_errors=True)
                    os.makedirs(folder_path, exist_ok=True)
                    print("Clear finished!")
                except:
                    print("Clear failed!")
                break
            else:
                print(f"Keep folder <{folder_path}>.")
                break


def check_path_exist(path, is_raise: bool = True, is_create: bool = False) -> bool:
    if os.path.exists(path):
        return True
    else:
        if is_create is True:
            res = input(f"Path <{path}> does not exist. Create it? ([Y]/n)")
            if res is None or res.lower() == "y":
                os.makedirs(path)
                return False
        if is_raise is True:
            raise FileNotFoundError(f"Path {path} does not exist.")
        return False


def save_as_pickle(data_to_save: typing.Any, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)


def read_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def path_join_output_folder(path: str):
    check_path_exist(settings.common_filepath.output_root_folder, is_raise=False, is_create=True)
    return os.path.join(settings.common_filepath.output_root_folder, path)


def path_join_trained_models_folder(path: str):
    check_path_exist(settings.train.trained_models_root_folder, is_raise=False, is_create=True)
    return os.path.join(settings.train.trained_models_root_folder, path)


def label_interceptor(label_list: list):
    new_label_list = []
    normal_amount = 0
    for label in label_list:
        if label == 0:
            normal_amount += 1
            new_label_list.append(0)
        else:
            # two class
            new_label_list.append(1)
        # else:
        #     # multi class
        #     new_label_list.append(label)
    print(
        f"Normal segments: {normal_amount}, abnormal segments: {len(label_list)-normal_amount}"
    )
    return new_label_list


def get_data_list_in_range(
    sorted_data_list: list, low_threshold: int, high_threshold: int
) -> list:
    """从已经升序排列的列表中根据上下阈值筛选出子列表

    Args:
        data_list (list): 升序排列的列表
        low_threshold (int): 下阈值
        high_threshold (int): 上阈值

    Returns:
        list: 子列表 ∈ [下阈值, 上阈值)
    """
    if low_threshold >= high_threshold:
        raise ValueError("high_threshold must be greater than low_threshold!")
    new_list = []
    start_index = bisect.bisect(sorted_data_list, low_threshold)
    if start_index > 0 and sorted_data_list[start_index - 1] == low_threshold:
        new_list.append(sorted_data_list[start_index - 1])
    for i in range(start_index, len(sorted_data_list)):
        if sorted_data_list[i] < high_threshold:
            new_list.append(sorted_data_list[i])
    return new_list


def plot_anything(
    data_to_plot: list,
    y_labels: list=None,
    block: bool = True,
    is_save=False,
    fig_name="fig.png",
    title: str = "Title",
) -> None:
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
    fig.suptitle(title)
    ax = fig.add_subplot(amount * 100 + 11)

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
    if y_labels is not None:
        ax.set_ylabel(y_labels[0])

    for index, each in enumerate(data[1:]):
        ax2 = fig.add_subplot(amount * 100 + 10 + index + 2, sharex=ax)
        if type(each) == tuple:
            if len(each) == 1:
                ax2.plot(each[0])
            elif len(each) == 2:
                ax2.plot(each[0], each[1])
        elif type(each) == list or type(each) == np.ndarray:
            ax2.plot(each)
        else:
            raise TypeError("things in data_to_plot must be tuple or list!")
        if y_labels is not None:
            ax2.set_ylabel(y_labels[index + 1])
    if block is True:
        plt.show()
    if is_save is True:
        plt.savefig(fig_name)


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)  # 选取小波函数
    a = data
    ca = []  # 近似分量
    cd = []  # 细节分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, "smooth")  # 进行5阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))  # 重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i == 3:
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
        ax.plot(y, "r")
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, "g")
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    plt.show()


def get_actual_sleep_duration_time(record_name: str, source_label_folder) -> int:
    # I think the actucal sleep duration time is the start time of the last WAKE stage
    # So we need to get the last WAKE stage time
    label_xml_filepath = os.path.join(source_label_folder, f"{record_name}-nsrr.xml")
    sleep_duration_time = -1
    try:
        with open(label_xml_filepath, "r", encoding="utf-8") as fr:
            all_lines = fr.readlines()
            raw_duration_line = all_lines[-5].strip()
            sleep_duration_time = int(float(raw_duration_line[7:-8]))
    finally:
        return sleep_duration_time


def transform_label_data_to_uniform_format(
    raw_sleep_apnea_label_data: typing.List[typing.Tuple[float, float, int, str]],
    sample_rate: int,
) -> typing.List[typing.Tuple[int, int, int, float]]:
    """_summary_

    Args:
        sleep_apnea_label_data (typing.List[typing.Tuple[float, float, int, str]]):
            [(start time, duration, event index, signal location), ...]
        sample_rate (int): _description_

    Returns:
        typing.List[typing.Tuple[int, int, int, float]]: _description_
    """
    uniform_format_label_data = []
    for start_sec, duration, event_index, _ in raw_sleep_apnea_label_data:
        uniform_format_label_data.append(
            (
                int(start_sec * sample_rate),
                int((start_sec + duration) * sample_rate),
                event_index,
                duration,
            )
        )
    return uniform_format_label_data


def get_sleep_apnea_label_list_according_to_data_length(
    uniform_format_sleep_apnea_label_data: typing.List[
        typing.Tuple[int, int, int, float]
    ],
    data_length: int,
) -> typing.List[int]:
    sleep_apnea_label_list = [0] * data_length
    for sleep_apnea_label in uniform_format_sleep_apnea_label_data:
        for i in range(sleep_apnea_label[0], min(data_length, sleep_apnea_label[1])):
            sleep_apnea_label_list[i] = sleep_apnea_label[2]
    return sleep_apnea_label_list
