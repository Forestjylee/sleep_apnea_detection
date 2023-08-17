import os
import pickle
import random
import traceback
import typing
from enum import Enum
from multiprocessing import Manager, Pool, cpu_count
from pprint import pprint

import numpy as np
from tqdm import tqdm

from config import settings
from utils import (check_path_exist, path_join_output_folder, read_pickle,
                   save_as_pickle)


class SleepApneaIntensity(Enum):
    Normal = 0
    Mild = 1
    Moderate = 2
    Severe = 3


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


def get_sleep_apnea_intensity_by_ahi(ahi) -> SleepApneaIntensity:
    """According to AASM
    0:正常Normal  1:轻度Mild  2:中度Moderate   3:重度Severe"""
    if ahi < 5:
        return SleepApneaIntensity.Normal
    elif ahi < 15:
        return SleepApneaIntensity.Mild
    elif ahi < 30:
        return SleepApneaIntensity.Moderate
    else:  # >=30
        return SleepApneaIntensity.Severe


def convert_sleep_apnea_intensity_dict_to_record_name_list(
    samples_SA_intensity: typing.Dict[SleepApneaIntensity, list]
) -> typing.List[str]:
    record_names = []
    for v in samples_SA_intensity.values():
        record_names.extend(v)
    return record_names


def get_sample_ahi(
    record_name: str, source_label_folder: str, sleep_apnea_label_folder: str
) -> float:
    """
    AHI(Apnea-Hypopnea Index) = (amount of apnea/hypopnea events) / sleep hours

    Returns:
        float: AHI
    """
    sleep_length_secs = get_actual_sleep_duration_time(record_name, source_label_folder)
    sa_label_path = os.path.join(
        sleep_apnea_label_folder, f"{record_name}_sa_events.pkl"
    )

    sleep_apnea_label_datas = read_pickle(sa_label_path)

    during_hour = sleep_length_secs / 3600
    ahi = len(sleep_apnea_label_datas) / during_hour
    return ahi


def _select_sample_according_to_rules(
    record_name: str,
    sensor_names_to_check: typing.List[str],
    raw_data_folder: str,
    source_label_folder: str,
    sleep_apnea_label_folder: str,
    results,
    errors,
) -> typing.Tuple[str, SleepApneaIntensity, int]:
    """
    Drop sample according to rules:
    1. Lack of sensor data
    2. Cannot get sleep duration time
    3. sleep duration time less than 6 hours

    Returns:
        SleepApneaIntensity: sleep apnea intensity
    """
    try:
        for sensor_name in sensor_names_to_check:
            raw_signal_filepath = os.path.join(
                raw_data_folder, f"{record_name}_{sensor_name}.pkl"
            )
            if not os.path.exists(raw_signal_filepath):
                return None

        sleep_length_secs = get_actual_sleep_duration_time(
            record_name, source_label_folder
        )
        if sleep_length_secs == -1:
            return None

        during_hour = sleep_length_secs / 3600
        if during_hour < 6:
            return None

        ahi = get_sample_ahi(record_name, source_label_folder, sleep_apnea_label_folder)
        sleep_apnea_intensity = get_sleep_apnea_intensity_by_ahi(ahi)

        results.put((record_name, sleep_apnea_intensity))

        return sleep_apnea_intensity
    except Exception as e:
        errors.put(record_name)


def select_samples_from_all(
    sensor_names_to_check: typing.List[str],
    raw_data_folder: str,
    source_label_folder: str,
    sleep_apnea_label_folder: str,
    is_multiprocess: bool = True,
) -> typing.Dict[SleepApneaIntensity, typing.List[str]]:
    check_path_exist(raw_data_folder)
    check_path_exist(source_label_folder)
    check_path_exist(sleep_apnea_label_folder)

    all_samples_SA_intensity = {
        SleepApneaIntensity.Normal: [],
        SleepApneaIntensity.Mild: [],
        SleepApneaIntensity.Moderate: [],
        SleepApneaIntensity.Severe: [],
    }

    if is_multiprocess is False:
        for sleep_apnea_label_file in tqdm(
            os.listdir(sleep_apnea_label_folder), desc="Selecting samples"
        ):
            if not sleep_apnea_label_file.endswith(".pkl"):
                continue

            record_name = sleep_apnea_label_file.split("_")[0]
            res = _select_sample_according_to_rules(
                record_name,
                sensor_names_to_check,
                raw_data_folder,
                source_label_folder,
                sleep_apnea_label_folder,
            )
            if res is not None:
                all_samples_SA_intensity[res].append(record_name)
    else:
        label_file_names = os.listdir(sleep_apnea_label_folder)
        pbar = tqdm(total=len(label_file_names))
        pbar.set_description("Selecting samples")
        update = lambda *args: pbar.update()
        with Pool(processes=cpu_count() - 1) as pool:
            m = Manager()
            errors = m.Queue()
            results = m.Queue()
            for label_file_name in label_file_names:
                record_name = label_file_name.split("_")[0]
                pool.apply_async(
                    _select_sample_according_to_rules,
                    args=(
                        record_name,
                        sensor_names_to_check,
                        raw_data_folder,
                        source_label_folder,
                        sleep_apnea_label_folder,
                        results,
                        errors,
                    ),
                    callback=update,
                )
            pool.close()
            pool.join()
        pbar.close()

        while not results.empty():
            tmp = results.get()
            all_samples_SA_intensity[tmp[1]].append(tmp[0])

        if not errors.empty():
            print(f"Error label file names: ")
            errs = []
            while not errors.empty():
                errs.append(errors.get())
            print(errs)
            print(f"There are {len(errs)} samples with errors.")
        else:
            print("All label files processed successfully!")

    return all_samples_SA_intensity


def samples_SA_statistics(
    samples_SA_intensity_filepath: str,
    source_label_folder: str,
    sleep_apnea_label_folder: str,
):
    samples_SA_intensity = read_pickle(samples_SA_intensity_filepath)
    available_samples_amount = 0
    for key in samples_SA_intensity.keys():
        print(f"key: {key.name}, amount: {len(samples_SA_intensity[key])}")
        ahis = []
        for record_name in tqdm(
            samples_SA_intensity[key], desc=f"Processing {key.name}"
        ):
            ahi = get_sample_ahi(
                record_name, source_label_folder, sleep_apnea_label_folder
            )
            ahis.append(ahi)
        ahi_array = np.array(ahis)
        print(f"Mean: {np.mean(ahi_array)}, std: {np.std(ahi_array)}")
        available_samples_amount += len(ahis)
    print(f"There are {available_samples_amount} samples available in total.")


def select_samples_in_specific_amount(
    samples_SA_intensity_path: str,
    samples_amount_dict: typing.Dict[SleepApneaIntensity, int],
    random_select: bool,
    random_seed: int = 42,
) -> typing.Dict[SleepApneaIntensity, list]:
    """根据SA严重程度比例挑选出样本

    Args:
        samples_amount_dict (typing.Dict[SleepApneaIntensity, int]): {
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
    check_path_exist(samples_SA_intensity_path)

    samples_SA_intensity = read_pickle(samples_SA_intensity_path)
    for key in tqdm(list(samples_SA_intensity.keys()), desc="Selecting samples"):
        if samples_amount_dict.get(key) is None:
            samples_SA_intensity[key] = []
            continue
        if len(samples_SA_intensity[key]) < samples_amount_dict[key]:
            raise ValueError(f"{key.name}类型样本的数量小于{samples_amount_dict[key]}")

        if random_select is True:
            random.seed(random_seed)
            samples_SA_intensity[key] = random.sample(
                samples_SA_intensity[key], samples_amount_dict[key]
            )
        else:
            samples_SA_intensity[key] = samples_SA_intensity[key][
                : samples_amount_dict[key]
            ]
    return samples_SA_intensity


if __name__ == "__main__":
    ## Some data storage path
    ## shhs1
    sensors_to_check = ["ABDO", "THOR", "NEW"]
    raw_data_folder = settings.shhs1.raw_data_path
    source_label_folder = settings.shhs1.source_sleep_apnea_label_path
    sleep_apnea_label_folder = settings.shhs1.sleep_apnea_label_path
    samples_SA_intensity_path = path_join_output_folder(
        settings.shhs1.samples_SA_intensity_info_filename
    )

    ## mesa
    # sensors_to_check = ["Abdo", "Thor", "Flow"]
    # raw_data_folder = settings.mesa.raw_data_path
    # source_label_folder = settings.mesa.source_sleep_apnea_label_path
    # sleep_apnea_label_folder = settings.mesa.sleep_apnea_label_path
    # samples_SA_intensity_path = path_join_output_folder(
    #     settings.mesa.samples_SA_intensity_info_filename
    # )

    ## Generate all samples' sleep apnea intensity info file
    ## If you have already generated the file, just comment the following lines
    all_samples_SA_intensity = select_samples_from_all(
        sensor_names_to_check=sensors_to_check,
        raw_data_folder=raw_data_folder,
        source_label_folder=source_label_folder,
        sleep_apnea_label_folder=sleep_apnea_label_folder,
    )
    if all_samples_SA_intensity:
        save_as_pickle(all_samples_SA_intensity, samples_SA_intensity_path)
        print(
            f"All samples' sleep apnea intensity info file saved to {samples_SA_intensity_path}"
        )

    ## Calculate some statistics of samples's sleep apnea info
    samples_SA_statistics(
        samples_SA_intensity_path, source_label_folder, sleep_apnea_label_folder
    )

    ## Select samples according to specific amount dict
    # Change the `samples_amount_dict` to what you need
    # samples_amount_dict = {
    #     # SleepApneaIntensity.Normal: 10,
    #     SleepApneaIntensity.Mild: 10,
    #     SleepApneaIntensity.Moderate: 10,
    #     SleepApneaIntensity.Severe: 10,
    # }
    # selected_samples = select_samples_in_specific_amount(
    #     samples_SA_intensity_path,
    #     samples_amount_dict,
    #     random_select=True,
    #     random_seed=42,
    # )
    # pprint(selected_samples)
    # pprint(convert_sleep_apnea_intensity_dict_to_record_name_list(selected_samples))
