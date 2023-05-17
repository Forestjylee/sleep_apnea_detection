import os
import pywt
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from select_samples import select_samples_in_specific_ratio, SleepApneaIntensity


from sleep_data_obj import SleepData
from utils import plot_anything, plot_signal_decomp


raw_signal_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\shhs1_respiration_signals"
SHHS_labels_source_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\AHI_source"
AHI_label_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\AHI_events"
target_train_data_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\train_data"
target_test_data_folder = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\test_data"
samples_SA_intensity_filepath = r"C:\Users\forestjylee\Developer\PythonProject\shhs_experiment\samples_SA_intensity.pkl"


def transfer_label_datas_to_target_format(label_datas: list, sample_rate: float) -> list:
    new_label_datas = []
    for start_sec, duration, event_index, signal_source_type in label_datas:
        # if signal_source_type != 'NEW':
        # 不保存根据鼻气流判断的呼吸暂停事件
        new_label_datas.append((
            int(start_sec * sample_rate),
            int((start_sec+duration) * sample_rate),
            event_index,
            duration
        ))
    return new_label_datas


def get_actual_sleep_duration_time(record_name: str):
    label_xml_filepath = os.path.join(SHHS_labels_source_folder, f"{record_name}-nsrr.xml")
    sleep_duration_time = 0
    try:
        with open(label_xml_filepath, 'r', encoding='utf-8') as fr:
            all_lines = fr.readlines()
            raw_duration_line = all_lines[-5].strip()
            sleep_duration_time = int(float(raw_duration_line[7:-8]))
    finally:
        return sleep_duration_time


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
    seg_len = int(5*sample_rate)
    for start_i in range(0, len(raw_datas), seg_len):
        new_seg = raw_datas[start_i:start_i+seg_len]
        # new_data.extend([np.max(new_seg)-np.min(new_seg)]*seg_len)
        if np.std(new_seg) < 0.02 and np.max(new_seg)-np.min(new_seg) < 0.05:
            new_data.extend([0]*seg_len)
        else:
            new_data.extend(raw_datas[start_i:start_i+seg_len])
        # for i in range(seg_len):
        #     if -0.03 < new_seg[i] < 0.03:
        #         new_seg[i] = 0
        # if new_seg.count(0) > seg_len * 0.8:
        #     new_data.extend([0]*seg_len)
        # else:
        #     new_data.extend(copied_data[start_i:start_i+seg_len])
    
    # 恢复零散的0片段
    seg_len_15s = int(60*sample_rate)
    start_i = 0
    while start_i < len(new_data):
        if new_data[start_i] != 0:
            start_i += 1
        else:
            temp_len = 0
            while start_i + temp_len < len(new_data) and new_data[start_i+temp_len] == 0:
                temp_len += 1
            if temp_len < seg_len_15s:
                for i in range(temp_len):
                    new_data[start_i+i] = raw_datas[start_i+i]
            start_i += temp_len
            
    # 合并零散的非0片段
    seg_len_5min = int(5*60*sample_rate)
    start_i = 0
    while start_i < len(new_data):
        if new_data[start_i] == 0:
            start_i += 1
        else:
            temp_len = 0
            while start_i + temp_len < len(new_data) and new_data[start_i+temp_len] != 0:
                temp_len += 1
            if temp_len < seg_len_5min:
                # 非0片段长度小于设定阈值，全赋0
                for i in range(temp_len):
                    new_data[start_i+i] = 0
            start_i += temp_len
            
    return new_data


def waveletDecompose(data, w='sym5', amount=3):
    w = pywt.Wavelet(w)#选取小波函数
    ca = []#近似分量
    a = data
    for i in range(amount):
        a, _ = pywt.dwt(a, w, 'smooth')#进行amount阶离散小波变换
        ca.append(a)

    rec_a = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构
    # print(type(rec_a[0]))   # type ndarray
    return rec_a


def single_process_shhs_data_for_train_test(record_name: str, sensor_name: str, target_save_folder: str):
    RE_SAMPLE_RATE = 10
    WINDOW_SIZE_SECS = 10     # 一个X样本的窗口大小
    SLIDE_STRIDE_SECS = 2     # 呼吸暂停片段滑动步长
    OVERLAP_LENGTH_SECS = 4      # 根据呼吸暂停标签,首个呼吸暂停事件长度 e.g.: [NNNNNNAAAA]

    raw_signal_filepath = os.path.join(
        raw_signal_folder, f"{record_name}_{sensor_name}.pkl")
    ahi_label_filepath = os.path.join(
        AHI_label_folder, f"{record_name}_AHI.pkl")
    if not os.path.exists(ahi_label_filepath):
        print(f"{record_name} donot have ahi labels!")

    X_data_to_train_or_test = []
    y_data_to_train_or_test = []

    with open(raw_signal_filepath, 'rb') as f:
        data_dict = pickle.load(f)
    raw_datas = data_dict["data"]
    sample_rate = int(data_dict["sample_rate"])
    data_length = int(get_actual_sleep_duration_time(record_name) * sample_rate) # 取医生标记的最后一个清醒片段的起始时间作为样本长度
    if len(raw_datas) < data_length:
        return
    else:
        raw_datas = raw_datas[:data_length]
        
    raw_datas = preprocess_raw_data(raw_datas, sample_rate)   # NEW AIR donot use

    sd = SleepData(sample_rate=sample_rate)
    sd.load_data_from_list(raw_datas).resample(RE_SAMPLE_RATE)
    # sd.z_score_normalize()   # NEW DATA use
    sd.filter(
        low_pass_cutoff=0.7,
        low_pass_filter_order=2,
        low_pass_filter_type='butter',
        # high_pass_cutoff=0.2,
        # high_pass_filter_order=2,
        # high_pass_filter_type='butter'
    )
    # sd.range_normalize(0, 1)
    normed_sd = sd.get_data()
    data_length = len(normed_sd)
    # normed_sd = waveletDecompose(normed_sd)[1].tolist()
    # plot_anything([raw_datas, waveletDecompose(normed_sd)[1]])

    with open(ahi_label_filepath, 'rb') as f:
        sleep_apnea_label_datas = pickle.load(f)
    sleep_apnea_label_datas = transfer_label_datas_to_target_format(
        sleep_apnea_label_datas, sd.sample_rate)
    # print(f"{record_name} apnea segments: {len(sleep_apnea_label_datas)}")

    # label_info index
    label_datas_index = 0
    label_datas_length = len(sleep_apnea_label_datas)

    segment_length_5s = int(5 * sd.sample_rate)
    segment_length = int(WINDOW_SIZE_SECS * sd.sample_rate)   # 窗口的长度
    start_index = int(180 * sd.sample_rate)     # 跳过开始时的不稳定片段（200s）
    slide_stride = int(SLIDE_STRIDE_SECS * sd.sample_rate)
    overlap_length = int(OVERLAP_LENGTH_SECS * sd.sample_rate)
    # pbar = tqdm(total=data_length)
    # pbar.update(start_index)
    while start_index < data_length:
        end_index = start_index + segment_length    # update end index
        prev_start_index = start_index
        if end_index >= data_length:
            break
        
        while label_datas_index < label_datas_length \
            and sleep_apnea_label_datas[label_datas_index][1] <= start_index:
            label_datas_index += 1
        
        if -0.001 <= normed_sd[end_index-1] <= 0.001 and -0.001 <= normed_sd[end_index-2] <= 0.001:
            start_index = end_index
            # pbar.update(segment_length)
            continue
        if -0.001 <= normed_sd[start_index] <= 0.001 and -0.001 <= normed_sd[start_index+1] <= 0.001:
            start_index += segment_length_5s
            # pbar.update(segment_length_5s)
            continue

        if label_datas_index == label_datas_length or end_index <= sleep_apnea_label_datas[label_datas_index][0]:
            normed_array = np.array(normed_sd[start_index:end_index])
            # normed_array = (normed_array + 1) / 2
            # normed_train_data = normed_array
            normed_train_data = sd.get_range_normalized_data(normed_array)
            # reca = waveletDecompose(normed_train_data)
            # plot_anything([normed_train_data, reca[0], reca[1], reca[2], reca[3], reca[4]])
            # plot_anything([normed_train_data])
            # X_data_to_train_or_test.append([reca[1]])
            X_data_to_train_or_test.append([normed_train_data])
            y_data_to_train_or_test.append(0)
            start_index += int(5*sd.sample_rate)   # set start index
            # start_index += segment_length
        elif end_index > sleep_apnea_label_datas[label_datas_index][0]:
            begin_start_index = sleep_apnea_label_datas[label_datas_index][0] + overlap_length - segment_length
            end_start_index = sleep_apnea_label_datas[label_datas_index][1] - overlap_length
            if end_start_index + segment_length > data_length:
                end_start_index = data_length - segment_length
            for temp_start in range(begin_start_index, end_start_index, slide_stride):
                if -0.001 <= normed_sd[temp_start+segment_length-1] <= 0.001:
                    temp_start += segment_length
                    continue
                if -0.001 <= normed_sd[temp_start] <= 0.001:
                    temp_start += slide_stride
                    continue
                normed_array = np.array(normed_sd[temp_start:temp_start+segment_length])
                # normed_array = (normed_array + 1) / 2
                # normed_train_data = normed_array
                normed_train_data = sd.get_range_normalized_data(normed_array)
                # plot_signal_decomp(normed_array, 'sym5', 'Respiratory')
                # reca = waveletDecompose(normed_train_data)
                # plot_anything([normed_train_data, reca[0], reca[1], reca[2], reca[3], reca[4]])
                # print(start_index, end_index, begin_start_index, temp_start)
                # plot_anything([normed_sd[temp_start:temp_start+segment_length]])
                # X_data_to_train_or_test.append([reca[1]])
                X_data_to_train_or_test.append([normed_train_data])
                y_data_to_train_or_test.append(
                    sleep_apnea_label_datas[label_datas_index][2])
            # set start index
            start_index = sleep_apnea_label_datas[label_datas_index][1]
            label_datas_index += 1
        # pbar.update(start_index-prev_start_index)
    # pbar.close()
    
    if len(y_data_to_train_or_test) < 1000:
        # 如果一个样本的有效片段少于1000个，直接不保存文件
        return

    X_data_to_train_or_test = np.array(
        X_data_to_train_or_test, dtype='float32').transpose((0, 2, 1))
    # if y_data_to_train_or_test.count(0) / len(y_data_to_train_or_test) > 0.90:
    #     # if normal segments account more than 17%, drop this sample
    #     return None
    # print(X_data_to_train_or_test)
    # print(len(X_data_to_train_or_test), len(y_data_to_train_or_test))

    if not os.path.exists(target_save_folder):
        os.makedirs(target_save_folder)
    with open(os.path.join(target_save_folder, f"{record_name}_{sensor_name}_X_data.pkl"), 'wb') as fw:
        pickle.dump(X_data_to_train_or_test, fw)
    with open(os.path.join(target_save_folder, f"{record_name}_{sensor_name}_y_data.pkl"), 'wb') as fw:
        pickle.dump(y_data_to_train_or_test, fw)
    # print(f"process {record_name}'s {sensor_name} data finished!")
    return record_name


def process_all_data(target_save_folder: str, sensor_name: str, start_index: int, end_index: int):
    print('<----------------------------------------', "Start extracting",
          '----------------------------------------------->')
    # clear folder
    shutil.rmtree(target_save_folder, ignore_errors=True)
    os.makedirs(target_save_folder, exist_ok=True)
    print("清空文件夹成功!")
    label_file_names = [label_file_name for label_file_name in os.listdir(
        raw_signal_folder) if sensor_name in label_file_name][start_index:end_index]
    # for label_file_name in tqdm(label_file_names, desc="Total processing"):
    #     record_name = label_file_name.split('_')[0]
    #     single_process_shhs_data_for_train_test(record_name, sensor_name, target_save_folder)
    with Pool(processes=cpu_count()-1) as pool:
        for label_file_name in label_file_names:
            record_name = label_file_name.split('_')[0]
            pool.apply_async(single_process_shhs_data_for_train_test, args=(
                record_name, sensor_name, target_save_folder, ))
        pool.close()
        pool.join()
    print('--------------------------------------------------',
          '---------------------------------------------------------------')


def process_selected_samples(target_save_folder: str, sensor_name: str):
    print('<----------------------------------------', "Start extracting",
          '----------------------------------------------->')
    samples_ratio_dict = {
        SleepApneaIntensity.Normal: 30,
        SleepApneaIntensity.Mild: 30,
        SleepApneaIntensity.Moderate: 30,
        SleepApneaIntensity.Severe: 40
    }
    selected_record_dict = select_samples_in_specific_ratio(
        samples_ratio_dict, sensor_name, False)

    # clear folder
    shutil.rmtree(target_save_folder, ignore_errors=True)
    os.makedirs(target_save_folder, exist_ok=True)
    print("清空文件夹成功!")

    for key, record_names in selected_record_dict.items():
        count = 0
        for record_name in record_names:
            try:
                if count == 20:
                    break
                single_process_shhs_data_for_train_test(
                    record_name, sensor_name, target_save_folder)
                count += 1
            except:
                print(key)

    # with Pool(processes=cpu_count()-1) as pool:
    #     for record_names in selected_record_dict.values():
    #         for record_name in record_names:
    #             pool.apply_async(single_process_shhs_data_for_train_test, args=(
    #                 record_name, sensor_name, target_save_folder, ))
    #     pool.close()
    #     pool.join()
    print('-----------------------------------------------------------------------------------------------------------------')


def process_samples_after_sift(target_save_folder: str, sensor_name: str, samples_SA_intensity_filepath: str):
    print('<----------------------------------------', "Start extracting",
          '----------------------------------------------->')
    # clear folder
    # shutil.rmtree(target_save_folder, ignore_errors=True)
    # os.makedirs(target_save_folder, exist_ok=True)
    # print("清空文件夹成功!")
    
    record_names = []
    with open(samples_SA_intensity_filepath, "rb") as f:
        samples_SA_intensity = pickle.load(f)
    for key in samples_SA_intensity.keys():
        record_names.extend(samples_SA_intensity[key])
    
    # for record_name in tqdm(record_names, desc="Total processing"):
    #     single_process_shhs_data_for_train_test(record_name, sensor_name, target_save_folder)
    
    # 多进程
    pbar = tqdm(total=len(record_names))
    pbar.set_description('Total processing')
    update = lambda *args: pbar.update()
    with Pool(processes=cpu_count()-2) as pool:
        for record_name in record_names:
            pool.apply_async(
                single_process_shhs_data_for_train_test, 
                args=(record_name, sensor_name, target_save_folder, ),
                callback=update,
                # error_callback=update
            )
        pool.close()
        pool.join()
    pbar.close()
        
    # 多线程
    # with ThreadPoolExecutor(max_workers=14) as w:
    #     futures = [
    #         w.submit(
    #             single_process_shhs_data_for_train_test, 
    #             record_name, sensor_name, target_save_folder) 
    #         for record_name in record_names
    #     ]
        
    #     for future in as_completed(futures):
    #         res = future.result()
    #         if res is not None: 
    #             print(f"{res} done")
    print('--------------------------------------------------',
          '---------------------------------------------------------------')


if __name__ == "__main__":
    sensor_name = "ABDO"      # ABDO / THOR / NEW（鼻气流）
    # single_process_shhs_data_for_train_test("shhs1-204302", sensor_name, target_train_data_folder)
    # process_all_data(target_train_data_folder, sensor_name, 0, 2400)
    # process_all_data(target_test_data_folder, sensor_name, 2400, 3000)
    # process_all_data(target_train_data_folder, sensor_name, 0, None)

    process_samples_after_sift(
        target_train_data_folder, sensor_name, samples_SA_intensity_filepath)
    # process_selected_samples(target_train_data_folder, sensor_name)
