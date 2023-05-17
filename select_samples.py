import os
import pickle
import typing
import random
import numpy as np
from enum import Enum
from tqdm import tqdm
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed


raw_signal_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\shhs1_respiration_signals"
AHI_source_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\AHI_source"
AHI_label_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\AHI_events"


SENSOR_NAMES = ["ABDO", "THOR", "NEW"]


class SleepApneaIntensity(Enum):
    Normal = 0
    Mild = 1
    Moderate = 2
    Severe = 3


def get_actual_sleep_duration_time(record_name: str) -> int:
    label_xml_filepath = os.path.join(AHI_source_folder, f"{record_name}-nsrr.xml")
    sleep_duration_time = 0
    try:
        with open(label_xml_filepath, 'r', encoding='utf-8') as fr:
            all_lines = fr.readlines()
            raw_duration_line = all_lines[-5].strip()
            sleep_duration_time = int(float(raw_duration_line[7:-8]))
    finally:
        return sleep_duration_time


def get_SA_intensity_by_ahi_amonut(ahi_amount) -> SleepApneaIntensity:
    """0:正常Normal  1:轻度Mild  2:中度Moderate   3:重度Severe"""
    if ahi_amount < 5:
        return SleepApneaIntensity.Normal
    elif ahi_amount < 15:
        return SleepApneaIntensity.Mild
    elif ahi_amount < 30:
        return SleepApneaIntensity.Moderate
    else:  # >=30
        return SleepApneaIntensity.Severe


def save_as_pickle(data_to_save: typing.Any, filepath: str) -> bool:
    """将任何Python数据结构以pickle的形式序列化存储到指定文件中

    Args:
        filepath (str): 目标文件地址

    Returns:
        bool: 是否保存成功
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)
        return True
    except Exception as e:
        return False


def read_pickle(filepath: str) -> None:
    if not os.path.exists(filepath):
        print(f"{filepath} is not exists.")
        return []

    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_sample_SA_intensity(record_name: str) -> typing.Tuple[str, SleepApneaIntensity, int]:
    """根据标签文件中SA事件的数量, 得到当前样本的SA严重程度
    1、只有包含所有SENSOR_NAMES中传感器原始数据的样本会被统计
    2、只有有效睡眠长度超过6小时的样本才被统计

    Args:
        record_name (str): _description_

    Returns:
        typing.Tuple[str, SleepApneaIntensity, int]: SA严重程度, ahi事件数量
    """
    for sensor_name in SENSOR_NAMES:
        raw_signal_filepath = os.path.join(raw_signal_folder, f"{record_name}_{sensor_name}.pkl")
        if not os.path.exists(raw_signal_filepath):
            print(f"{record_name} donot have raw {sensor_name} signal data!")
            return None
    sleep_length_secs = get_actual_sleep_duration_time(record_name)
    if sleep_length_secs == -1:
        return None
    ahi_label_filepath = os.path.join(AHI_label_folder, f"{record_name}_AHI.pkl")
    if not os.path.exists(ahi_label_filepath):
        print(f"{record_name} donot have ahi labels!")
        return None
    
    during_hour = sleep_length_secs / 3600   # 数据长度
    if during_hour < 6:  # 判断数据是否超过6个小时
        return None
    
    with open(ahi_label_filepath, 'rb') as f:
        sleep_apnea_label_datas = pickle.load(f)
    # for start_sec, duration, event_index, signal_source_type in sleep_apnea_label_datas:
    #     pass
    ahi_amount = len(sleep_apnea_label_datas) / during_hour
    return record_name, get_SA_intensity_by_ahi_amonut(ahi_amount), ahi_amount


def classify_all_samples(AHI_label_folder: str, start_index: int=0):
    if not os.path.exists(AHI_label_folder):
        print(f"{AHI_label_folder}标签文件夹路径不存在")
        return
    
    all_samples_SA_intensity = {
        SleepApneaIntensity.Normal: [],
        SleepApneaIntensity.Mild: [],
        SleepApneaIntensity.Moderate: [],
        SleepApneaIntensity.Severe: []
    }
    
    label_file_names = [label_file_name for label_file_name in os.listdir(
        AHI_source_folder)][start_index:]
    
    with ThreadPoolExecutor(max_workers=12) as w:
        futures = [
            w.submit(
                get_sample_SA_intensity, 
                f"{label_file_name.split('-')[0]}-{label_file_name.split('-')[1]}") 
            for label_file_name in label_file_names
        ]
        
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                all_samples_SA_intensity[res[1]].append(res[0])
                print(f"{res[0]} done")
    
    for key in all_samples_SA_intensity.keys():
        print(f"key: {key.name}, amount: {len(all_samples_SA_intensity[key])}")
    
    save_res = save_as_pickle(all_samples_SA_intensity, f"samples_SA_intensity.pkl")
    if save_res is True:
        print("保存样本呼吸暂停严重情况成功!")
    else:
        print("保存样本呼吸暂停严重情况失败")
        
        
def AHI_amount_statistics(samples_SA_intensity_filepath: str):
    with open(samples_SA_intensity_filepath, "rb") as f:
        samples_SA_intensity = pickle.load(f)
    for key in samples_SA_intensity.keys():
        print(f"key: {key.name}, amount: {len(samples_SA_intensity[key])}")
        ahi_amounts = []
        for record_name in tqdm(samples_SA_intensity[key], desc=f"Processing {key.name}"):
            _, _, ahi_amount = get_sample_SA_intensity(record_name)
            ahi_amounts.append(ahi_amount)
        ahi_array = np.array(ahi_amounts)
        print(f"Mean: {np.mean(ahi_array)}, std: {np.std(ahi_array)}")
        


def select_samples_in_specific_ratio(
    samples_ratio_dict: typing.Dict[SleepApneaIntensity, int], random_selection: bool
) -> typing.Dict[SleepApneaIntensity, list]:
    """根据SA严重程度比例挑选出样本

    Args:
        samples_ratio_dict (typing.Dict[SleepApneaIntensity, int]): {
            SleepApneaIntensity.Normal: 250,
            SleepApneaIntensity.Mild: 250,
            SleepApneaIntensity.Moderate: 250,
            SleepApneaIntensity.Severe: 250
        }
        random_selection (bool): 是否随机选取

    Returns:
        dict: {
            SleepApneaIntensity.Normal: [...],
            SleepApneaIntensity.Mild: [...],
            SleepApneaIntensity.Moderate: [...],
            SleepApneaIntensity.Severe: [...]
        }
    """
    samples_SA_intensity_filepath = f"samples_SA_intensity.pkl"
    if not os.path.exists(samples_SA_intensity_filepath):
        print(f"SA intensity file has not been generated.")
        return []

    with open(samples_SA_intensity_filepath, "rb") as f:
        samples_SA_intensity = pickle.load(f)
    print("Selecting samples...")
    for key in samples_SA_intensity.keys():
        if samples_ratio_dict.get(key) is None:
            samples_SA_intensity[key] = []
            continue
        if len(samples_SA_intensity[key]) < samples_ratio_dict[key]:
            print(f"{key.name}类型样本的数量小于{samples_ratio_dict[key]}")
            return []
        if random_selection is True:
            samples_SA_intensity[key] = random.sample(samples_SA_intensity[key], samples_ratio_dict[key])
        else:
            samples_SA_intensity[key] = samples_SA_intensity[key][:samples_ratio_dict[key]]
    return samples_SA_intensity


if __name__ == "__main__":
    # run this firstly
    # classify_all_samples(AHI_label_folder, start_index=0)
    AHI_amount_statistics("samples_SA_intensity.pkl")

    # samples_ratio_dict = {
    #     # SleepApneaIntensity.Normal: 50,
    #     # SleepApneaIntensity.Mild: 50,
    #     # SleepApneaIntensity.Moderate: 50,
    #     SleepApneaIntensity.Severe: 50
    # }
    # sensor_name = "ABDO"    # ABDO | THOR | NEW
    # samples_SA_intensity = read_pickle(f"{sensor_name}_samples_SA_intensity.pkl")
    # for key in samples_SA_intensity.keys():
    #     print(f"key: {key.name}, amount: {len(samples_SA_intensity[key])}")
    # res = select_samples_in_specific_ratio(samples_ratio_dict, sensor_name, random_selection=False)
    # print(res)
