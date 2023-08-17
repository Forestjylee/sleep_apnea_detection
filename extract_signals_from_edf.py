from alive_progress import alive_bar
import os
import pickle
import typing
import pyedflib
import traceback
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool, cpu_count, Manager

from config import settings
from utils import check_path_exist


def get_edf_file_info(edf_filepath, is_print: bool = True) -> dict:
    edf_file = pyedflib.EdfReader(edf_filepath)
    n = edf_file.signals_in_file
    signal_labels = edf_file.getSignalLabels()
    signal_headers = edf_file.getSignalHeaders()

    if is_print is True:
        print(f"signal numbers: {n}")
        pprint(f"Labels: {signal_labels}")
        pprint(signal_headers)
    return {
        "signal_numbers": n,
        "signal_labels": signal_labels,
        "signal_headers": signal_headers,
    }


def read_edf_file(
    edf_filepath: str, signal_channel_names_to_save: list
) -> typing.List[dict]:
    # every signal channel to be saved must exist, or it will return []
    try:
        edf_file = pyedflib.EdfReader(edf_filepath)
        n = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        signal_headers = edf_file.getSignalHeaders()

        # get channel indexes we need
        channel_indexes_we_need = []
        for channel_name in signal_channel_names_to_save:
            channel_indexes_we_need.append(signal_labels.index(channel_name))

        results = []
        # read channels we need
        for i in channel_indexes_we_need:
            data_to_save = {}
            data_to_save["data"] = edf_file.readSignal(i)
            results.append(data_to_save)
        return results
    except:
        traceback.print_exc()
        return []


def _shhs1_read_edf_file_and_save_raw_signal_data(
    edf_file_name: str,
    source_edf_folder: str,
    target_raw_data_path: str,
    signal_channel_names_to_save: list,
    error_edf_names,
):
    try:
        record_name = edf_file_name.split(".")[0]
        edf_filepath = os.path.join(source_edf_folder, edf_file_name)
        edf_file = pyedflib.EdfReader(edf_filepath)
        n = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        signal_headers = edf_file.getSignalHeaders()

        # get channel indexes we need
        channel_indexes_we_need = []
        for channel_name in signal_channel_names_to_save:
            channel_indexes_we_need.append(signal_labels.index(channel_name))

        # read channels we need
        for i in channel_indexes_we_need:
            data_to_save = {}
            data_to_save["sample_rate"] = signal_headers[i]["sample_rate"]
            data_to_save["data"] = edf_file.readSignal(i)
            with open(
                os.path.join(
                    target_raw_data_path,
                    f"{record_name}_{signal_channel_names_to_save[channel_indexes_we_need.index(i)].split()[0]}.pkl",
                ),
                "wb",
            ) as fw:
                pickle.dump(data_to_save, fw)
    except Exception as e:
        error_edf_names.put({"filename": edf_file_name, "err_msg": repr(e)})
        raise e


def shhs1_process_all_edf_files():
    source_edf_folder = settings.shhs1.source_edf_path
    target_raw_signal_data_folder = settings.shhs1.raw_data_path
    signal_channel_names_to_save = ["THOR RES", "ABDO RES", "NEW AIR"]
    
    check_path_exist(source_edf_folder)
    check_path_exist(target_raw_signal_data_folder, is_create=True)

    error_edf_names = Manager().Queue()
    edf_file_names = os.listdir(source_edf_folder)
    with alive_bar(len(edf_file_names), title="Processing edf files") as pbar:
        update = lambda *args: pbar()
        with Pool(processes=cpu_count() - 1) as pool:
            for edf_file_name in edf_file_names:
                pool.apply_async(
                    _shhs1_read_edf_file_and_save_raw_signal_data,
                    args=(
                        edf_file_name,
                        source_edf_folder,
                        target_raw_signal_data_folder,
                        signal_channel_names_to_save,
                        error_edf_names,
                    ),
                    callback=update,
                )
            pool.close()
            pool.join()
    
    print(f"Error edf file names: ")
    errs = []
    while not error_edf_names.empty():
        errs.append(error_edf_names.get())
    pprint(errs)


def _mesa_read_edf_file_and_save_raw_signal_data(
    edf_file_name: str,
    source_edf_folder: str,
    target_raw_data_path: str,
    signal_channel_names_to_save: list,
    error_edf_names, 
):
    try:
        record_name = edf_file_name.split('.')[0]
        edf_filepath = os.path.join(source_edf_folder, edf_file_name)
        edf_file = pyedflib.EdfReader(edf_filepath)
        n = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        signal_headers = edf_file.getSignalHeaders()
        
        # get channel indexes we need
        channel_indexes_we_need = []
        for channel_name in signal_channel_names_to_save:
            channel_indexes_we_need.append(signal_labels.index(channel_name))
        
        # read channels we need
        for i in channel_indexes_we_need:
            data_to_save = {}
            data_to_save["sample_rate"] = signal_headers[i]["sample_rate"]
            data_to_save["data"] = edf_file.readSignal(i)
            with open(os.path.join(target_raw_data_path, f"{record_name}_{signal_channel_names_to_save[channel_indexes_we_need.index(i)]}.pkl"), 'wb') as fw:
                pickle.dump(data_to_save, fw)
    except Exception as e:
        error_edf_names.put({"filename": edf_file_name, "err_msg": repr(e)})
        raise e


def mesa_process_all_edf_files():
    source_edf_folder = settings.mesa.source_edf_path
    target_raw_signal_data_folder = settings.mesa.raw_data_path
    signal_channel_names_to_save = ["SpO2", "Abdo", "Thor", "Flow"]
    
    check_path_exist(source_edf_folder)
    check_path_exist(target_raw_signal_data_folder, is_create=True)

    error_edf_names = Manager().Queue()
    edf_file_names = os.listdir(source_edf_folder)
    with alive_bar(len(edf_file_names), title="Processing edf files") as pbar:
        update = lambda *args: pbar()
        with Pool(processes=cpu_count() - 1) as pool:
            for edf_file_name in edf_file_names:
                pool.apply_async(
                    _mesa_read_edf_file_and_save_raw_signal_data,
                    args=(
                        edf_file_name,
                        source_edf_folder,
                        target_raw_signal_data_folder,
                        signal_channel_names_to_save,
                        error_edf_names,
                    ),
                    callback=update,
                )
            pool.close()
            pool.join()
    
    print(f"Error edf file names: ")
    errs = []
    while not error_edf_names.empty():
        errs.append(error_edf_names.get())
    pprint(errs)


if __name__ == "__main__":
    # Read a single EDF file
    # test_edf_filepath = r"C:\Users\forestjylee\Developer\PythonProject\sleep_apnea_detection\data\shhs1\source_edfs\shhs1-200002.edf"
    # get_edf_file_info(test_edf_filepath)
    # data = read_edf_file(test_edf_filepath, ["THOR RES", "ABDO RES", "NEW AIR"])
    
    # Read dataset SHHS1's EDF files
    shhs1_process_all_edf_files()
    
    # Read dataset MESA's EDF files
    # mesa_process_all_edf_files()
