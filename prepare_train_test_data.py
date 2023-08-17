import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import Manager, Pool, cpu_count
from pprint import pprint

import numpy as np
import pywt
from alive_progress import alive_bar
from tqdm import tqdm

from config import settings
from select_samples import SleepApneaIntensity
from sleep_data_obj import SleepData
from utils import (check_path_exist, clear_folder,
                   get_actual_sleep_duration_time, path_join_output_folder,
                   plot_anything, save_as_pickle,
                   transform_label_data_to_uniform_format)


def get_record_names_from_SA_intensity_file(samples_SA_intensity_filepath: str):
    record_names = []
    with open(samples_SA_intensity_filepath, "rb") as f:
        samples_SA_intensity = pickle.load(f)
    for key in samples_SA_intensity.keys():
        record_names.extend(samples_SA_intensity[key])
    return record_names


def generate_train_validate_test_SA_intensity_file(
    all_samples_SA_intensity_filepath: str, 
    random_seed: int=42,
    train_validate_test_ratio_dict: dict={"train": 10, "test": 0, "validate": 0}
) -> typing.Tuple[str, str, str]:
    """generate three SA intensity file (train, validate, test) according to all samples SA intensity file

    Args:
        all_samples_SA_intensity_filepath (str): all available samples info
        train_validate_test_ratio_dict (_type_, optional): sum is 10. Defaults to {"train": 10, "test": 0, "validate": 0}.
    """
    random.seed(random_seed)
    
    with open(all_samples_SA_intensity_filepath, "rb") as f:
        samples_SA_intensity = pickle.load(f)
    
    train_samples_SA_intensity, validate_samples_SA_intensity, test_samples_SA_intensity = {}, {}, {}
    for key in samples_SA_intensity.keys():
        train_offset = int(train_validate_test_ratio_dict["train"] / 10 * len(samples_SA_intensity[key]))
        validate_offset = int(train_validate_test_ratio_dict["validate"] / 10 * len(samples_SA_intensity[key]))
        test_offset = int(train_validate_test_ratio_dict["test"] / 10 * len(samples_SA_intensity[key]))
        temp_record_names = samples_SA_intensity[key]
        random.shuffle(temp_record_names)
        train_samples_SA_intensity[key] = temp_record_names[:train_offset]
        validate_samples_SA_intensity[key] = temp_record_names[train_offset:train_offset+validate_offset]
        test_samples_SA_intensity[key] = temp_record_names[train_offset+validate_offset:train_offset+validate_offset+test_offset]
    
    folder_path = os.path.dirname(all_samples_SA_intensity_filepath)
    filename = os.path.basename(all_samples_SA_intensity_filepath)

    train_samples_SA_intensity_filapth = os.path.join(folder_path, f"train_{filename}")
    validate_samples_SA_intensity_filapth = os.path.join(folder_path, f"validate_{filename}")
    test_samples_SA_intensity_filepath = os.path.join(folder_path, f"test_{filename}")
    
    save_as_pickle(train_samples_SA_intensity, train_samples_SA_intensity_filapth)
    save_as_pickle(validate_samples_SA_intensity, validate_samples_SA_intensity_filapth)
    save_as_pickle(test_samples_SA_intensity, test_samples_SA_intensity_filepath)
    
    return train_samples_SA_intensity_filapth, validate_samples_SA_intensity_filapth, test_samples_SA_intensity_filepath


def preprocess_raw_data(raw_datas, sample_rate) -> list:
    raw_datas = np.array(raw_datas)
    raw_datas = raw_datas - np.mean(raw_datas)
    raw_datas = raw_datas.tolist()

    # 移除带有上下限溢出(方波)的片段, 连续3个点值相同视为方波
    # seg_len = int(5*sample_rate)
    # for start_i in range(0, len(raw_datas), seg_len):
    #     for i in range(start_i+2, start_i+seg_len):
    #         if (raw_datas[i] < -0.95 or raw_datas[i] > 0.95) \
    #             and raw_datas[i] == raw_datas[i-1] \
    #                 and raw_datas[i] == raw_datas[i-2]:
    #             for j in range(seg_len):
    #                 raw_datas[start_i+j] = 0
    #             break

    # 移除值为0的附近的无信号片段
    # new_data = copied_data
    new_data = []
    seg_len = int(5 * sample_rate)
    for start_i in range(0, len(raw_datas), seg_len):
        new_seg = raw_datas[start_i : start_i + seg_len]
        # new_data.extend([np.max(new_seg)-np.min(new_seg)]*seg_len)
        if np.std(new_seg) < 0.02 and np.max(new_seg) - np.min(new_seg) < 0.05:
            new_data.extend([0] * seg_len)
        else:
            new_data.extend(raw_datas[start_i : start_i + seg_len])
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

    return new_data


def waveletDecompose(data, w="sym5", amount=3):
    w = pywt.Wavelet(w)  # 选取小波函数
    ca = []  # 近似分量
    a = data
    for i in range(amount):
        a, _ = pywt.dwt(a, w, "smooth")  # 进行amount阶离散小波变换
        ca.append(a)

    rec_a = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))  # 重构
    # print(type(rec_a[0]))   # type ndarray
    return rec_a


def single_process_shhs_mesa_data_for_train_test(
    record_name: str,
    sensor_name: str,
    raw_data_folder: str,
    source_label_folder: str,
    sleep_apnea_folder: str,
    target_save_folder: str,
    error_record_names=None
):
    try:
        # 数据的目标采样率（通过重采样实现）
        resample_rate = settings.preprocess.resample_rate
        # 一个X样本的窗口大小
        window_length_secs = settings.preprocess.window_length_secs
        # 呼吸暂停片段滑动步长
        window_slide_stride_secs = settings.preprocess.window_slide_stride_secs
        # 根据呼吸暂停标签,首个呼吸暂停事件长度 e.g.: [NNNNNNAAAA]
        sleep_apnea_overlap_length_secs = settings.preprocess.sleep_apnea_overlap_length_secs

        raw_data_path = os.path.join(raw_data_folder, f"{record_name}_{sensor_name}.pkl")
        sleep_apnea_label_path = os.path.join(
            sleep_apnea_folder, f"{record_name}_sa_events.pkl"
        )

        with open(raw_data_path, "rb") as f:
            data_dict = pickle.load(f)
        raw_datas = data_dict["data"]
        sample_rate = int(data_dict["sample_rate"])
        data_length = int(
            get_actual_sleep_duration_time(record_name, source_label_folder) * sample_rate
        )  # 取医生标记的最后一个清醒片段的起始时间作为样本长度
        if len(raw_datas) < data_length:
            return
        else:
            raw_datas = raw_datas[:data_length]

        raw_datas = preprocess_raw_data(raw_datas, sample_rate)  # NEW AIR donot use

        sd = SleepData(sample_rate=sample_rate)
        sd.load_data_from_list(raw_datas).resample(resample_rate)
        sd.z_score_normalize()
        sd.filter(
            low_pass_cutoff=settings.preprocess.low_pass_cutoff,
            low_pass_filter_order=settings.preprocess.low_pass_filter_order,
            low_pass_filter_type="butter",
            # high_pass_cutoff=settings.preprocess.high_pass_cutoff,
            # high_pass_filter_order=settings.preprocess_high_pass_filter_order,
            # high_pass_filter_type='butter'
        )

        normed_sd = sd.get_data()
        data_length = len(normed_sd)

        with open(sleep_apnea_label_path, "rb") as f:
            sleep_apnea_label_datas = pickle.load(f)
        uniform_format_sleep_apnea_label_data = transform_label_data_to_uniform_format(
            sleep_apnea_label_datas, sd.sample_rate
        )
        # print(f"{record_name} apnea segments: {len(uniform_format_sleep_apnea_label_data)}")

        X_data_to_train_or_test = []
        y_data_to_train_or_test = []
        # label_info index
        label_datas_index = 0
        label_datas_length = len(uniform_format_sleep_apnea_label_data)

        segment_length_5s = int(5 * sd.sample_rate)
        segment_length = int(window_length_secs * sd.sample_rate)  # Length of sliding window
        start_index = int(180 * sd.sample_rate)  # Skip beginning unstable segments（180s）
        slide_stride = int(window_slide_stride_secs * sd.sample_rate)
        overlap_length = int(sleep_apnea_overlap_length_secs * sd.sample_rate)
        
        while start_index < data_length:
            end_index = start_index + segment_length  # update end index
            if end_index >= data_length:
                break

            while (
                label_datas_index < label_datas_length
                and uniform_format_sleep_apnea_label_data[label_datas_index][1] <= start_index
            ):
                label_datas_index += 1

            if (
                -0.001 <= normed_sd[end_index - 1] <= 0.001
                and -0.001 <= normed_sd[end_index - 2] <= 0.001
            ):
                start_index = end_index
                continue
            if (
                -0.001 <= normed_sd[start_index] <= 0.001
                and -0.001 <= normed_sd[start_index + 1] <= 0.001
            ):
                start_index += segment_length_5s
                continue

            if (
                label_datas_index == label_datas_length
                or end_index <= uniform_format_sleep_apnea_label_data[label_datas_index][0]
            ):
                normed_array = np.array(normed_sd[start_index:end_index])
                
                # normed_train_data = sd.get_range_normalized_data(normed_array)
                normed_train_data = normed_array.tolist()
                
                X_data_to_train_or_test.append([normed_train_data])
                y_data_to_train_or_test.append(0)
                start_index += segment_length_5s  # set start index
                # start_index += segment_length
            elif end_index > uniform_format_sleep_apnea_label_data[label_datas_index][0]:
                begin_start_index = (
                    uniform_format_sleep_apnea_label_data[label_datas_index][0]
                    + overlap_length
                    - segment_length
                )
                end_start_index = (
                    uniform_format_sleep_apnea_label_data[label_datas_index][1] - overlap_length
                )
                if end_start_index + segment_length > data_length:
                    end_start_index = data_length - segment_length
                for temp_start in range(begin_start_index, end_start_index, slide_stride):
                    if -0.001 <= normed_sd[temp_start + segment_length - 1] <= 0.001:
                        temp_start += segment_length
                        continue
                    if -0.001 <= normed_sd[temp_start] <= 0.001:
                        temp_start += slide_stride
                        continue
                    normed_array = np.array(
                        normed_sd[temp_start : temp_start + segment_length]
                    )

                    # normed_train_data = sd.get_range_normalized_data(normed_array)
                    normed_train_data = normed_array.tolist()
                    
                    X_data_to_train_or_test.append([normed_train_data])
                    y_data_to_train_or_test.append(
                        uniform_format_sleep_apnea_label_data[label_datas_index][2]
                    )
                # set start index
                start_index = uniform_format_sleep_apnea_label_data[label_datas_index][1]
                label_datas_index += 1

        if len(y_data_to_train_or_test) < 1000:
            # If a sample's valid segment less than 1000, don't save that sample
            return

        X_data_to_train_or_test = np.array(
            X_data_to_train_or_test, dtype="float32"
        ).transpose((0, 2, 1))
        # if y_data_to_train_or_test.count(0) / len(y_data_to_train_or_test) > 0.90:
        #     # if normal segments account more than 17%, drop this sample
        #     return None
        # print(X_data_to_train_or_test)
        # print(len(X_data_to_train_or_test), len(y_data_to_train_or_test))

        if not os.path.exists(target_save_folder):
            os.makedirs(target_save_folder)
        with open(
            os.path.join(target_save_folder, f"{record_name}_{sensor_name}_X_data.pkl"),
            "wb",
        ) as fw:
            pickle.dump(X_data_to_train_or_test, fw)
        with open(
            os.path.join(target_save_folder, f"{record_name}_{sensor_name}_y_data.pkl"),
            "wb",
        ) as fw:
            pickle.dump(y_data_to_train_or_test, fw)
        # print(f"process {record_name}'s {sensor_name} data finished!")
        return record_name
    except Exception as e:
        error_record_names.put({"record_name": record_name, "err_msg": repr(e)})
        raise e


def process_samples_after_sift(
    raw_data_folder: str,
    source_label_folder: str,
    sleep_apnea_folder: str,
    target_save_folder: str,
    sensor_name: str,
    samples_SA_intensity_filepath: str,
):
    print(
        "<----------------------------------------",
        "Start extracting",
        "----------------------------------------------->",
    )
    # Ask user to clear folder
    clear_folder(target_save_folder)

    record_names = get_record_names_from_SA_intensity_file(samples_SA_intensity_filepath)

    ## Single process
    # for record_name in tqdm(record_names, desc="Total processing"):
    #     single_process_shhs_mesa_data_for_train_test(
    #         record_name,
    #         sensor_name,
    #         raw_data_folder,
    #         source_label_folder,
    #         sleep_apnea_folder,
    #         target_save_folder,
    #     )

    ## Multi-process
    error_record_names = Manager().Queue()
    with alive_bar(len(record_names), title="Processing samples") as pbar:
        update = lambda *args: pbar()
        with Pool(processes=cpu_count() - 2) as pool:
            for record_name in record_names:
                pool.apply_async(
                    single_process_shhs_mesa_data_for_train_test,
                    args=(
                        record_name,
                        sensor_name, 
                        raw_data_folder,
                        source_label_folder,
                        sleep_apnea_folder,
                        target_save_folder,
                        error_record_names
                    ),
                    callback=update,
                    # error_callback=update
                )
            pool.close()
            pool.join()
    print(f"Error record names: ")
    errs = []
    while not error_record_names.empty():
        errs.append(error_record_names.get())
    pprint(errs)
    
    print(
        "--------------------------------------------------",
        "---------------------------------------------------------------",
    )


if __name__ == "__main__":
    # shhs1
    sensor_name = "ABDO"  # ABDO / THOR / NEW（鼻气流）
    raw_data_folder = settings.shhs1.raw_data_path
    source_label_folder = settings.shhs1.source_sleep_apnea_label_path
    sleep_apnea_label_folder = settings.shhs1.sleep_apnea_label_path
    target_train_data_folder = settings.shhs1.train_data_path
    target_validation_data_folder = settings.shhs1.validation_data_path
    target_test_data_folder = settings.shhs1.test_data_path
    all_samples_SA_intensity_filepath = path_join_output_folder(
        settings.shhs1.samples_SA_intensity_info_filename
    )

    # mesa
    # sensor_name = "Abdo"  # Abdo / Thor / Flow（鼻气流）
    # raw_data_folder = settings.mesa.raw_data_path
    # source_label_folder = settings.mesa.source_sleep_apnea_label_path
    # sleep_apnea_label_folder = settings.mesa.sleep_apnea_label_path
    # target_train_data_folder = settings.mesa.train_data_path
    # target_validation_data_folder = settings.mesa.validation_data_path
    # target_test_data_folder = settings.mesa.test_data_path
    # all_samples_SA_intensity_filepath = path_join_output_folder(
    #     settings.mesa.samples_SA_intensity_info_filename
    # )

    check_path_exist(raw_data_folder, is_raise=True, is_create=False)
    check_path_exist(source_label_folder, is_raise=True, is_create=False)
    check_path_exist(sleep_apnea_label_folder, is_raise=True, is_create=False)

    check_path_exist(target_train_data_folder, is_raise=False, is_create=True)
    check_path_exist(target_validation_data_folder, is_raise=False, is_create=True)
    check_path_exist(target_test_data_folder, is_raise=False, is_create=True)
    
    train_samples_SA_intensity_filepath, validate_samples_SA_intensity_filepath, test_samples_SA_intensity_filepath = \
    generate_train_validate_test_SA_intensity_file(
        all_samples_SA_intensity_filepath,
        random_seed=42,
        train_validate_test_ratio_dict={"train":7, "test": 2, "validate": 1}
    )

    ## Process one
    # single_process_shhs_mesa_data_for_train_test(
    #     "shhs1-204302",
    #     sensor_name,
    #     raw_data_folder,
    #     source_label_folder,
    #     sleep_apnea_label_folder,
    #     target_train_data_folder,
    # )

    ## Multi process
    ## Process all samples
    # process_samples_after_sift(
    #     raw_data_folder,
    #     source_label_folder,
    #     sleep_apnea_label_folder,
    #     target_train_data_folder,
    #     sensor_name,
    #     all_samples_SA_intensity_filepath,
    # )
    ## Process train, validation, test samples seperately
    process_samples_after_sift(
        raw_data_folder,
        source_label_folder,
        sleep_apnea_label_folder,
        target_train_data_folder,
        sensor_name,
        train_samples_SA_intensity_filepath,
    )
    process_samples_after_sift(
        raw_data_folder,
        source_label_folder,
        sleep_apnea_label_folder,
        target_validation_data_folder,
        sensor_name,
        validate_samples_SA_intensity_filepath,
    )
    process_samples_after_sift(
        raw_data_folder,
        source_label_folder,
        sleep_apnea_label_folder,
        target_test_data_folder,
        sensor_name,
        test_samples_SA_intensity_filepath,
    )
