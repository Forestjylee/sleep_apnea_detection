import os
import glob
import pickle
import typing
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pywt
from kymatio.numpy import Scattering1D
from tqdm import tqdm

from config import settings
from display_data import plot_sensor_and_label_data_use_matplotlib
from select_samples import SleepApneaIntensity
from sleep_data_obj import SleepData
from utils import (
    check_path_exist,
    get_actual_sleep_duration_time,
    get_sleep_apnea_label_list_according_to_data_length,
    path_join_output_folder,
    plot_anything,
    read_pickle,
    transform_label_data_to_uniform_format,
)


def test_data_preprocess_method(record_name: str, sensor_name: str):
    RE_SAMPLE_RATE = 10
    WINDOW_SIZE_SECS = 10  # 一个X样本的窗口大小
    SLIDE_STRIDE_SECS = 2  # 呼吸暂停片段滑动步长
    OVERLAP_LENGTH_SECS = 4  # 根据呼吸暂停标签,首个呼吸暂停事件长度 e.g.: [NNNNNNAAAA]

    raw_signal_filepath = os.path.join(
        raw_signal_folder, f"{record_name}_{sensor_name}.pkl"
    )
    ahi_label_filepath = os.path.join(AHI_label_folder, f"{record_name}_AHI.pkl")
    if not os.path.exists(ahi_label_filepath):
        print(f"{record_name} donot have ahi labels!")

    X_data_to_train_or_test = []
    y_data_to_train_or_test = []

    # 取医生标记的最后一个清醒片段的起始时间作为样本长度

    with open(raw_signal_filepath, "rb") as f:
        data_dict = pickle.load(f)
    raw_datas = data_dict["data"]
    sample_rate = data_dict["sample_rate"]
    data_length = int(get_actual_sleep_duration_time(record_name) * sample_rate)
    if len(raw_datas) < data_length:
        return
    else:
        raw_datas = raw_datas[:data_length]

    raw_datas = raw_datas - np.mean(raw_datas)
    raw_datas = raw_datas.tolist()

    copied_data = deepcopy(raw_datas)
    # 移除带有上下限溢出(方波)的片段, 连续3个点值相同视为方波
    # seg_len = int(5*sample_rate)
    # for start_i in range(0, len(copied_data), seg_len):
    #     for i in range(start_i+2, start_i+seg_len):
    #         if (copied_data[i] < -0.95 or copied_data[i] > 0.95) \
    #             and copied_data[i] == copied_data[i-1] \
    #                 and copied_data[i] == copied_data[i-2]:
    #             for j in range(seg_len):
    #                 copied_data[start_i+j] = 0
    #             break

    # 移除值为0的附近的无信号片段
    # new_data = copied_data
    new_data = []
    seg_len = int(5 * sample_rate)
    for start_i in range(0, len(copied_data), seg_len):
        new_seg = copied_data[start_i : start_i + seg_len]
        # new_data.extend([np.max(new_seg)-np.min(new_seg)]*seg_len)
        if np.std(new_seg) < 0.02 and np.max(new_seg) - np.min(new_seg) < 0.05:
            new_data.extend([0] * seg_len)
        else:
            new_data.extend(copied_data[start_i : start_i + seg_len])
        # for i in range(seg_len):
        #     if -0.03 < new_seg[i] < 0.03:
        #         new_seg[i] = 0
        # if new_seg.count(0) > seg_len * 0.8:
        #     new_data.extend([0]*seg_len)
        # else:
        #     new_data.extend(copied_data[start_i:start_i+seg_len])

    # 恢复零散的0片段
    seg_len_15s = int(60 * sample_rate)
    start_i = 0
    while start_i < len(new_data):
        if new_data[start_i] != 0:
            start_i += 1
        else:
            temp_len = 0
            while (
                start_i + temp_len < len(new_data) and new_data[start_i + temp_len] == 0
            ):
                temp_len += 1
            if temp_len < seg_len_15s:
                for i in range(temp_len):
                    new_data[start_i + i] = raw_datas[start_i + i]
            start_i += temp_len

    # 合并零散的非0片段
    seg_len_5min = int(5 * 60 * sample_rate)
    start_i = 0
    while start_i < len(new_data):
        if new_data[start_i] == 0:
            start_i += 1
        else:
            temp_len = 0
            while (
                start_i + temp_len < len(new_data) and new_data[start_i + temp_len] != 0
            ):
                temp_len += 1
            if temp_len < seg_len_5min:
                # 非0片段长度小于设定阈值，全赋0
                for i in range(temp_len):
                    new_data[start_i + i] = 0
            start_i += temp_len

    # plot_anything([raw_datas, new_data])
    # return
    # plot_signal_decomp(new_data[10000:10100], 'sym5', 'Respiratory')

    sd = SleepData(sample_rate=sample_rate)
    sd.load_data_from_list(new_data).resample(RE_SAMPLE_RATE)
    # sd.z_score_normalize()
    sd.filter(
        low_pass_cutoff=0.7,
        low_pass_filter_order=4,
        low_pass_filter_type="butter",
        # high_pass_cutoff=0.2,
        # high_pass_filter_order=2,
        # high_pass_filter_type='butter'
    )

    with open(ahi_label_filepath, "rb") as f:
        sleep_apnea_label_datas = pickle.load(f)
    sleep_apnea_label_datas = transfer_label_datas_to_target_format(
        sleep_apnea_label_datas, sd.sample_rate
    )
    sleep_apnea_label_list = [0] * sd.get_data_length()
    # sleep_apnea_label_list = [0] * len(raw_datas)
    for train_index_label in sleep_apnea_label_datas:
        if train_index_label[1] > sd.get_data_length():
            for i in range(train_index_label[0], sd.get_data_length()):
                sleep_apnea_label_list[i] = train_index_label[2]
            break
        for i in range(train_index_label[0], train_index_label[1]):
            sleep_apnea_label_list[i] = train_index_label[2]
    # print(f"{record_name} apnea segments: {len(sleep_apnea_label_datas)}")
    plot_anything([raw_datas, sd.get_data(), sleep_apnea_label_list])


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


if __name__ == "__main__":
    raw_data_folder = settings.shhs1_raw_data_path
    sleep_apnea_label_folder = settings.shhs1_sleep_apnea_label_path

    sensor_names = ["ABDO", "THOR", "NEW"]
    record_name = "shhs1-200001"
    sample_rate = settings.sample_rate

    # shhs1-200015 weak signal
    for record_file in glob.glob(os.path.join(raw_data_folder, f"*_ABDO.pkl")):
        record_name = record_file.split("\\")[-1].split("_")[0]
        print(record_name)
        plot_sensor_and_label_data_use_matplotlib(
            sensor_names=sensor_names,
            record_name=record_name,
            sample_rate=sample_rate,
            raw_data_folder=raw_data_folder,
            sleep_apnea_label_folder=sleep_apnea_label_folder,
            title=f"{record_name}'s sensor data and label data",
        )
