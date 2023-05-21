import os
import pickle
import typing
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pywt
from kymatio.numpy import Scattering1D
from tqdm import tqdm

from config import settings
from select_samples import SleepApneaIntensity
from sleep_data_obj import SleepData
from utils import plot_anything, path_join_output_folder, check_path_exist


def transfer_label_data_to_uniform_format(
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


def get_spectrogram_x_y(data: list, sample_rate: float):
    x, y = SleepData(sample_rate).load_data_from_list(data).get_fft_frequency_spectrum()
    # plt.plot(x, y)
    # plt.show()
    return x, y


def get_actual_sleep_duration_time(record_name: str, source_label_folder) -> int:
    # I think the actucal sleep duration time is the start time of last WAKE stage
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


def plot_aligned_raw_data_and_label(
    record_name: str,
    sensor_name: str,
    raw_data_folder: str,
    sleep_apnea_label_folder: str,
):
    raw_data_filepath = os.path.join(
        raw_data_folder, f"{record_name}_{sensor_name}.pkl"
    )
    sa_label_filepath = os.path.join(
        sleep_apnea_label_folder, f"{record_name}_sa_events.pkl"
    )
    
    check_path_exist(raw_data_filepath)
    check_path_exist(sa_label_filepath)

    with open(raw_data_filepath, "rb") as f:
        data_dict = pickle.load(f)
    raw_data = data_dict["data"]
    sample_rate = data_dict["sample_rate"]

    with open(sa_label_filepath, "rb") as f:
        raw_sleep_apnea_label_data = pickle.load(f)

    uniform_format_sleep_apnea_label_data = transfer_label_data_to_uniform_format(
        raw_sleep_apnea_label_data, sample_rate
    )
    sleep_apnea_label_list = get_sleep_apnea_label_list_according_to_data_length(
        uniform_format_sleep_apnea_label_data, len(raw_data)
    )

    # print(f"{record_name} apnea segments: {len(raw_sleep_apnea_label_data)}")
    plot_anything([raw_data, sleep_apnea_label_list], title="Aligned raw data and label")


def single_process_shhs_data_for_train_test(record_name: str, sensor_name: str):
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

    with open(raw_signal_filepath, "rb") as f:
        data_dict = pickle.load(f)
    raw_datas = data_dict["data"]
    sample_rate = data_dict["sample_rate"]
    print(sample_rate)

    # 取医生标记的最后一个清醒片段的起始时间作为样本长度
    data_length = get_actual_sleep_duration_time(record_name) * RE_SAMPLE_RATE

    sd = SleepData(sample_rate=sample_rate)
    sd.load_data_from_list(raw_datas)
    sd.resample(RE_SAMPLE_RATE)
    # q1, q2, q3 = sd.get_quartile_values(sd.get_data())
    # max_threshold = q1 - 1.5 * (q3 - q1)
    # min_threshold = q3 + 1.5 * (q3 - q1)
    # plt.plot(sd.get_data())
    # plt.plot([max_threshold]*data_length)
    # plt.plot([min_threshold]*data_length)
    # plt.plot([0.06]*data_length)
    # plt.plot([-0.06]*data_length)
    # plt.show()
    sd.remove_mean()
    # sd.filter(
    #     low_pass_cutoff=0.6,
    #     low_pass_filter_order=4,
    #     low_pass_filter_type='butter',
    #     high_pass_cutoff=0.2,
    #     high_pass_filter_order=2,
    #     high_pass_filter_type='butter'
    # )
    sd.slice(0, data_length)
    # normed_sd = SleepData.get_adaptive_window_normalized_data(sd.get_data(), int(WINDOW_SIZE_SECS*sd.sample_rate), 0.5, 0.25)
    normed_sd = sd.get_data()
    # data_length = len(normed_sd)     # 取有数据显示的长度作为总长度

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
    plot_anything([normed_sd, sleep_apnea_label_list])
    return

    # label_info index
    label_datas_index = 0
    label_datas_length = len(sleep_apnea_label_datas)

    segment_length = int(WINDOW_SIZE_SECS * sd.sample_rate)  # 窗口的长度
    start_index = int(180 * sd.sample_rate)  # 跳过开始时的不稳定片段（200s）
    slide_stride = int(SLIDE_STRIDE_SECS * sd.sample_rate)
    overlap_length = int(OVERLAP_LENGTH_SECS * sd.sample_rate)
    pbar = tqdm(total=data_length)
    pbar.update(start_index)
    end_index = start_index + segment_length
    while start_index < data_length:
        prev_start_index = start_index
        if end_index > data_length:
            break

        if (
            label_datas_index == label_datas_length
            or end_index <= sleep_apnea_label_datas[label_datas_index][0]
        ):
            # normed_train_data = SleepData.get_range_normalized_data(
            #     normed_sd[start_index:end_index])
            # normed_train_data = SleepData.get_z_score_normalized_data(
            #     normed_sd[start_index:end_index])
            normed_train_data = normed_sd[start_index:end_index]
            X_data_to_train_or_test.append([normed_train_data])
            y_data_to_train_or_test.append(0)
            start_index += segment_length // 2  # set start index
            # x, y = get_spectrogram_x_y(normed_train_data, RE_SAMPLE_RATE)
            # plot_anything([normed_train_data, (x, y)])
            # start_index += slide_stride
        elif end_index > sleep_apnea_label_datas[label_datas_index][0]:
            begin_start_index = (
                sleep_apnea_label_datas[label_datas_index][0]
                + overlap_length
                - segment_length
            )
            end_start_index = (
                sleep_apnea_label_datas[label_datas_index][1] - overlap_length
            )
            if end_start_index + segment_length > data_length:
                end_start_index = data_length - segment_length
            for temp_start in range(begin_start_index, end_start_index, slide_stride):
                # normed_train_data = SleepData.get_range_normalized_data(
                #     normed_sd[temp_start:temp_start+segment_length])
                # normed_train_data = SleepData.get_z_score_normalized_data(
                #     normed_sd[temp_start:temp_start+segment_length])
                normed_train_data = normed_sd[temp_start : temp_start + segment_length]
                X_data_to_train_or_test.append([normed_train_data])
                y_data_to_train_or_test.append(
                    sleep_apnea_label_datas[label_datas_index][2]
                )
                # x, y = get_spectrogram_x_y(normed_train_data, RE_SAMPLE_RATE)
                # plot_anything([normed_train_data, (x, y)])
                # plot_anything([normed_train_data])
            # set start index
            start_index = sleep_apnea_label_datas[label_datas_index][1]
            label_datas_index += 1
        pbar.update(start_index - prev_start_index)
        end_index = start_index + segment_length  # update end index
    pbar.close()

    # X_data_to_train_or_test = np.array(
    #     X_data_to_train_or_test, dtype='float32').transpose((0, 2, 1))
    # if y_data_to_train_or_test.count(0) / len(y_data_to_train_or_test) > 0.90:
    #     # if normal segments account more than 17%, drop this sample
    #     return None
    # print(X_data_to_train_or_test)
    # print(len(X_data_to_train_or_test), len(y_data_to_train_or_test))

    print(f"process {record_name}'s {sensor_name} data finished!")
    return record_name


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
    source_label_folder = settings.shhs1_source_sleep_apnea_label_path
    sleep_apnea_label_folder = settings.shhs1_sleep_apnea_label_path
    samples_SA_intensity_filepath = path_join_output_folder(
        "shhs1_all_samples_SA_intensity_info.pkl"
    )
    
    record_name = "shhs1-200001"
    sensor_name = "ABDO"  # ABDO / THOR / NEW（鼻气流
    plot_aligned_raw_data_and_label(record_name, sensor_name, raw_data_folder, sleep_apnea_label_folder)
