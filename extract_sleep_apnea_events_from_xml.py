import os
import typing
import pickle
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool, cpu_count, Manager

from config import settings
from utils import check_path_exist


def save_label_data(record_name: str, target_label_folder: str, label_datas: list):
    with open(
        os.path.join(target_label_folder, f"{record_name}_sa_events.pkl"), "wb"
    ) as f:
        pickle.dump(label_datas, f)


def _shhs1_read_and_save_sleep_apnea_label_data(
    source_label_path: str,
    source_label_folder: str,
    target_label_folder: str,
    error_label_files,
):
    events_index_mapper = {
        "Hypopnea": 1,
        "Obstructive apnea": 2,
        "Central apnea": 3,
    }
    event_names = events_index_mapper.keys()
    try:
        record_name = source_label_path.split(".")[0][:-5]
        source_label_path = os.path.join(source_label_folder, source_label_path)
        with open(source_label_path, "r", encoding="utf-8") as fr:
            sleep_apnea_label_datas = []
            all_lines = fr.readlines()
            line_index = 1
            while line_index < len(all_lines):
                for event_name in event_names:
                    if event_name in all_lines[line_index]:
                        sleep_apnea_label_datas.append(
                            (
                                float(all_lines[line_index + 1].strip()[7:-8]),       # start time
                                float(all_lines[line_index + 2].strip()[10:-11]),     # duration
                                events_index_mapper[event_name],                      # event type
                                all_lines[line_index + 3].strip()[16:-17].split()[0], # signal location (according to)
                            )
                        )
                        line_index += 2
                        break
                line_index += 1
        save_label_data(record_name, target_label_folder, sleep_apnea_label_datas)
    except Exception as e:
        error_label_files.put(source_label_path)


def shhs1_process_all_label_files():
    error_label_files = Manager().Queue()
    source_label_folder = settings.shhs1_source_sleep_apnea_label_path
    target_label_folder = settings.shhs1_sleep_apnea_label_path

    check_path_exist(source_label_folder)
    check_path_exist(target_label_folder, is_create=True)

    label_file_names = os.listdir(source_label_folder)
    pbar = tqdm(total=len(label_file_names))
    pbar.set_description("Processing label files")
    update = lambda *args: pbar.update()
    with Pool(processes=cpu_count() - 1) as pool:
        for label_file_name in label_file_names:
            pool.apply_async(
                _shhs1_read_and_save_sleep_apnea_label_data,
                args=(
                    label_file_name,
                    source_label_folder,
                    target_label_folder,
                    error_label_files,
                ),
                callback=update,
            )
        pool.close()
        pool.join()
    pbar.close()

    if not error_label_files.empty():
        print(f"Error label file names: ")
        errs = []
        while not error_label_files.empty():
            errs.append(error_label_files.get())
        pprint(errs)
    else:
        print("All label files processed successfully!")


def _mesa_read_and_save_sleep_apnea_label_datas(
    source_label_path: str,
    source_label_folder: str,
    target_label_folder: str,
    error_label_files,
):
    events_index_mapper = {
        "Hypopnea": 1,
        "Obstructive apnea": 2,
        "Central apnea": 3,
    }
    event_names = events_index_mapper.keys()
    try:
        record_name = source_label_path.split(".")[0][:-5]
        source_label_path = os.path.join(source_label_folder, source_label_path)
        if os.path.isfile(source_label_path) is False:
            raise ValueError("请传入有效的文件路径")
        with open(source_label_path, "r", encoding="utf-8") as fr:
            sleep_apnea_label_datas = []
            all_lines = fr.readlines()
            line_index = 1
            while line_index < len(all_lines):
                for event_name in event_names:
                    if event_name in all_lines[line_index]:
                        sleep_apnea_label_datas.append(
                            (
                                float(all_lines[line_index + 1].strip()[7:-8]),
                                float(all_lines[line_index + 2].strip()[10:-11]),
                                events_index_mapper[event_name],
                                all_lines[line_index + 3].strip()[16:-17].split()[0],
                            )
                        )
                        line_index += 2
                        break
                line_index += 1
        save_label_data(record_name, target_label_folder, sleep_apnea_label_datas)
    except Exception as e:
        error_label_files.put(source_label_path)


def mesa_process_all_label_files():
    error_label_files = Manager().Queue()
    source_label_folder = settings.mesa_source_sleep_apnea_label_path
    target_label_folder = settings.mesa_sleep_apnea_label_path

    check_path_exist(source_label_folder)
    check_path_exist(target_label_folder, is_create=True)

    label_file_names = os.listdir(source_label_folder)
    pbar = tqdm(total=len(label_file_names))
    pbar.set_description("Processing label files")
    update = lambda *args: pbar.update()
    with Pool(processes=cpu_count() - 1) as pool:
        for label_file_name in label_file_names:
            pool.apply_async(
                _mesa_read_and_save_sleep_apnea_label_datas,
                args=(
                    label_file_name,
                    source_label_folder,
                    target_label_folder,
                    error_label_files,
                ),
                callback=update,
            )
        pool.close()
        pool.join()
    pbar.close()

    if not error_label_files.empty():
        print(f"Error label file names: ")
        errs = []
        while not error_label_files.empty():
            errs.append(error_label_files.get())
        pprint(errs)
    else:
        print("All label files processed successfully!")


if __name__ == "__main__":
    # read_and_save_sleep_apnea_label_datas(r"E:\datasets\shhs\polysomnography\annotations-events-nsrr\shhs1\shhs1-200001-nsrr.xml")
    shhs1_process_all_label_files()
    # mesa_process_all_label_files()
