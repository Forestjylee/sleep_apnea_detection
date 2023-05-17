'''
sleep data analysis object
@author: Junyi
@createDate: 07/18/21
@updateDate: 05/11/22
'''
import math
import typing
import numpy as np
from tqdm import tqdm
from scipy import signal
from typing import Union
from copy import deepcopy
from functools import wraps
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle


# define annotation type
SD = typing.TypeVar('SD', bound='SleepData')


class SleepData(object):
    """
    WARNING:
    所有以get开头的函数不会改变自身的self.data
    除此之外的函数均会改变自身的self.data
    =======================================

    示例：
    >>> s = SleepData(sample_rate=200)
    >>> s.load_data_from_filepath("test.txt")
    >>> s[500:10500].remove_mean().filter(
            high_pass_cutoff=1.5,
            high_pass_filter_order=2,
            high_pass_filter_type='butter'
        ).plot()
    """

    def __init__(self, sample_rate: Union[int, float], lite_mode: bool = True):
        """[summary]

        Args:
            sample_rate (Union[int, float]): 采样率
            lite_mode (bool, optional): 是否保存原始数据的副本，默认不保存. Defaults to True.
        """
        self.sample_rate = sample_rate
        self.lite_mode = lite_mode

        self.data = []
        self.__raw_data = []    # 原始数据，永不变动
        self.__data_length = 0
        self.__raw_data_length = 0

    def __str__(self):
        return f"<SleepData object at {hex(id(self))}, raw data length is {self.__raw_data_length}, now data length is {self.__data_length}>"

    def __repr__(self):
        return f"<SleepData object at {hex(id(self))}, raw data length is {self.__raw_data_length}, now data length is {self.__data_length}>"

    def __len__(self):
        return self.__data_length

    def __add__(self, other):
        if isinstance(other, (int, float)):
            for index in range(self.__data_length):
                self.data[index] += other
            return self
        elif isinstance(other, list):
            if self.__data_length == len(other):
                for index, each in enumerate(other):
                    self.data[index] += each
            else:
                raise ValueError(
                    f"The length of list add to SleepData object must equal to the length of SleepData")
        else:
            raise TypeError(f"Not support type {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            for index in range(self.__data_length):
                self.data[index] -= other
            return self
        elif isinstance(other, list):
            if self.__data_length == len(other):
                for index, each in enumerate(other):
                    self.data[index] -= each
            else:
                raise ValueError(
                    f"The length of list substract to SleepData object must equal to the length of SleepData")
        else:
            raise TypeError(f"Not support type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            for index in range(self.__data_length):
                self.data[index] *= other
            return self
        elif isinstance(other, list):
            if self.__data_length == len(other):
                for index, each in enumerate(other):
                    self.data[index] *= each
            else:
                raise ValueError(
                    f"The length of list multiply to SleepData object must equal to the length of SleepData")
        else:
            raise TypeError(f"Not support type {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            for index in range(self.__data_length):
                self.data[index] /= other
            return self
        elif isinstance(other, list):
            if self.__data_length == len(other):
                for index, each in enumerate(other):
                    self.data[index] /= each
            else:
                raise ValueError(
                    f"The length of list div to SleepData object must equal to the length of SleepData")
        else:
            raise TypeError(f"Not support type {type(other)}")

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            for index in range(self.__data_length):
                self.data[index] //= other
            return self
        elif isinstance(other, list):
            if self.__data_length == len(other):
                for index, each in enumerate(other):
                    self.data[index] //= each
            else:
                raise ValueError(
                    f"The length of list floor div to SleepData object must equal to the length of SleepData")
        else:
            raise TypeError(f"Not support type {type(other)}")

    def __iter__(self):
        """enable use `for` to walk through self.data"""
        return iter(self.data)

    def __getitem__(self, index):
        """define `s[::] | s[] | s[[...]]` operations"""

        if isinstance(index, int) or isinstance(index, np.int64):
            return self.data[index]
        elif isinstance(index, slice):
            return self.get_sliced_data(index.start, index.stop, index.step)
        elif isinstance(index, list):
            return get_SleepData_instance(self.sample_rate).load_data_from_list(self.get_data_values_from_indexes(index))
        else:
            raise TypeError(f"Not support type {type(index)}")

    def __setitem__(self, index, value):
        if isinstance(index, int) or isinstance(index, np.int64):
            self.data[index] = value
        elif isinstance(index, list):
            for i, each_index in index:
                self.data[each_index] = value[i]

    def __handle_data_enplace(func):
        """decorator in class template"""
        @wraps(func)
        def inner(self, *args, **kwargs) -> Union[typing.Any, SD]:
            enplace = self.__default_enplace if kwargs.get(
                'enplace', None) is None else kwargs['enplace']
            ret = func(self, *args, **kwargs)
            if enplace is True:
                self.set_data(ret)
                return self
            return ret
        return inner

    def load_data_from_filepath(self, filepath: str, column: int, to_which_type: type = int) -> SD:
        """从哪一列开始读取数据，从0开始"""
        self.data = read_sleep_data_column(
            filepath, column, to_which_type=to_which_type)
        self.__data_length = len(self.data)
        if self.lite_mode is False:
            self.__raw_data = deepcopy(self.data)
            self.__raw_data_length = self.__data_length
        return self

    def load_data_from_list(self, raw_data: list) -> SD:
        if isinstance(raw_data, list):
            self.data = deepcopy(raw_data)
        elif isinstance(raw_data, np.ndarray):
            self.data = raw_data.tolist()
        else:
            raise TypeError("Only list and ndarray are valid!")
        self.__data_length = len(raw_data)
        if self.lite_mode is False:
            self.__raw_data = deepcopy(raw_data)
            self.__raw_data_length = self.__data_length
        return self

    def set_sample_rate(self, sample_rate: float) -> SD:
        self.sample_rate = sample_rate
        return self

    def set_data(self, data: Union[list, SD, np.ndarray]) -> SD:
        if isinstance(data, list):
            self.data = deepcopy(data)
            self.__data_length = len(data)
        elif isinstance(data, SleepData):
            self.data = data.get_data()
            self.__data_length = data.get_data_length()
        elif isinstance(data, np.ndarray):
            self.data = data.tolist()
            self.__data_length = data.size
        else:
            raise TypeError(f"set_data() donot support type {type(data)}")
        return self

    def get_data(self, start: int = None, end: int = None) -> list:
        if start is None:
            start = 0
        if end is None:
            end = self.__data_length
        return self.data[start:end]

    def get_data_length(self) -> int:
        return self.__data_length

    def reset_data(self) -> SD:
        self.data = self.set_data(self.__raw_data)
        return self

    def set_data_as_raw_data(self) -> SD:
        self.__raw_data = self.get_data()
        return self

    def get_raw_data(self) -> list:
        return deepcopy(self.__raw_data)

    def get_raw_data_length(self) -> int:
        return self.__raw_data_length

    def to_list(self) -> list:
        return deepcopy(self.data)

    def to_nparray(self) -> np.array:
        return np.array(self.data, dtype=np.float64)

    def copy(self) -> SD:
        s = get_SleepData_instance(self.sample_rate)
        return s.load_data_from_list(self.data)

    def plot(
        self, start: int = 0, end: int = 0,
        with_raw_data: bool = False, exit_after_plot: bool = False
    ) -> None:
        end = len(self.data) if end == 0 else end
        if with_raw_data is True:
            plt.plot(self.__raw_data)
        plt.plot(self.data[start:end])
        plt.show()
        if exit_after_plot is True:
            exit(0)

    def slice(self, start: int = None, end: int = None, step: int = None) -> SD:
        self.set_data(self.data[start:end:step])
        return self

    def resample(self, new_sample_rate: float) -> SD:
        """进行数据重采样，使用scipy.signal.resample函数"""
        secs = self.__data_length / self.sample_rate
        self.set_data(signal.resample(self.get_data(),
                                      int(secs*new_sample_rate)).tolist())
        self.sample_rate = new_sample_rate
        return self

    def process_with_func(self, my_function, args: tuple = (), kwargs: dict = {}) -> SD:
        """
        用自定义函数处理数据，自定义的函数至少接收一个list类型的数据，并返回一个list类型的数据
        自定义函数的参数传递可以使用args(元组)，kwargs(字典)传入
        """
        processed_data = my_function(self.get_data(), *args, **kwargs)
        self.set_data(processed_data)
        return self

    def filter(
        self, high_pass_cutoff: float = -1, low_pass_cutoff: float = -1,
        high_pass_filter_order: int = 4, high_pass_filter_type: str = 'butter',
        low_pass_filter_order: int = 4, low_pass_filter_type: str = 'butter',
    ) -> SD:
        """
        获取原始波形滤波之后的波形
        支持带通、低通、高通
        支持两种滤波器
        butter means butterworth
        bessel means bessel
        低通则high_pass_cutoff=-1,高通low_pass_cutoff=-1
        """
        filter_data = self.data
        if high_pass_cutoff != -1:
            filter_data = iir_high_pass_filter(
                filter_data, self.sample_rate, high_pass_cutoff, high_pass_filter_order, high_pass_filter_type)
        if low_pass_cutoff != -1:
            filter_data = iir_low_pass_filter(
                filter_data, self.sample_rate, low_pass_cutoff, low_pass_filter_order, low_pass_filter_type)
        self.set_data(filter_data.tolist())
        return self

    def analysis_bcg_wavelet(self, peak_list: typing.List[int], one_side_window_size_sec: float):
        """将BCG中所有峰值点的波形片段集中展示

        Args:
            peak_list (typing.List[int]): 峰值点序列
            one_side_window_size_sec (float): 窗口的一半大小(单位是秒)
        """
        one_side_window_length = one_side_window_size_sec * self.sample_rate
        windows_list = []
        for peak in peak_list:
            left = int(peak - one_side_window_length)
            right = int(peak + one_side_window_length)
            if left >= 0 and right <= self.__data_length:
                windows_list.append([left, right])

        for window in windows_list:
            plt.plot([y for y in self.__range_normalize(
                self.get_data(window[0], window[1]), -1, 1)])
        plt.show()

    def chebyshev_type1_filter(self, ftype: str, freqs: list = [], order: int = 5, rp: int = 3) -> SD:
        """切比雪夫1型滤波器
        ===============
        usually used to remove baseline wandering and power line interference

        Args:
            ftype (str): 滤波器类型['low_pass' or 'high_pass' or 'band_pass' or 'bacdstop']
            freqs (list, optional): cutoff frequencies. Defaults to [].
            order (int, optional): 滤波器阶数. Defaults to 5.
            rp (int, optional): 切比雪夫滤波器中的必设参数. Defaults to 3.

        Returns:
            SD: SleepData instance
        """

        filter_data = self.get_data()
        nyq = 0.5 * self.sample_rate

        if ftype == 'low_pass':
            assert len(freqs) == 1
            cut = freqs[0]/nyq
            b, a = signal.cheby1(order, rp, cut, btype='lowpass')
        elif ftype == 'high_pass':
            assert len(freqs) == 1
            cut = freqs[0]/nyq
            b, a = signal.cheby1(order, rp, cut, btype='highpass')
        elif ftype == 'band_pass':
            assert len(freqs) == 2
            lowcut, highcut = freqs[0]/nyq, freqs[1]/nyq
            b, a = signal.cheby1(
                order, rp, [lowcut, highcut], btype='bandpass')
        elif ftype == 'band_stop':
            assert len(freqs) == 2
            lowcut, highcut = freqs[0]/nyq, freqs[1]/nyq
            b, a = signal.cheby1(
                order, rp, [lowcut, highcut], btype='bandstop')

        self.set_data(signal.lfilter(b, a, filter_data).tolist())
        return self

    def chebyshev_type2_filter(self, ftype: str, freqs: list = [], order: int = 5, rc: int = 40) -> SD:
        """切比雪夫2型滤波器
        ===============
        usually used to remove baseline wandering and power line interference

        Args:
            ftype (str): 滤波器类型['low_pass' or 'high_pass' or 'band_pass' or 'bacdstop']
            freqs (list, optional): cutoff frequencies. Defaults to [].
            order (int, optional): 滤波器阶数. Defaults to 5.
            rc (int, optional): 切比雪夫滤波器中的必设参数. Defaults to 40.

        Returns:
            SD: SleepData instance
        """

        filter_data = self.get_data()
        nyq = 0.5 * self.sample_rate

        if ftype == 'low_pass':
            assert len(freqs) == 1
            cut = freqs[0]/nyq
            b, a = signal.cheby2(order, rc, cut, btype='lowpass')
        elif ftype == 'high_pass':
            assert len(freqs) == 1
            cut = freqs[0]/nyq
            b, a = signal.cheby2(order, rc, cut, btype='highpass')
        elif ftype == 'band_pass':
            assert len(freqs) == 2
            lowcut, highcut = freqs[0]/nyq, freqs[1]/nyq
            b, a = signal.cheby2(
                order, rc, [lowcut, highcut], btype='bandpass')
        elif ftype == 'band_stop':
            assert len(freqs) == 2
            lowcut, highcut = freqs[0]/nyq, freqs[1]/nyq
            b, a = signal.cheby2(
                order, rc, [lowcut, highcut], btype='bandstop')

        self.set_data(signal.lfilter(b, a, filter_data).tolist())
        return self

    def notch_filter(self, cutoff: float, Q: int = 30) -> SD:
        """陷波滤波器
        ============
        used for power line interference

        Args:
            cutoff (float): 目标截断频率
            Q (int, optional): Quality factor. Defaults to 30.

        Returns:
            SD: SleepData instance
        """
        filter_data = self.get_data()
        nyq = 0.5 * self.sample_rate
        b, a = signal.iirnotch(cutoff/nyq, Q)
        self.set_data(signal.lfilter(b, a, filter_data).tolist())
        return self

    def scale(self, scale: float) -> SD:
        """对data进行乘法缩放操作"""
        scaled_data = [i * scale for i in self.data]
        self.set_data(scaled_data)
        return self

    def opposite_value(self) -> SD:
        """以0为对称轴，对所有的数取相反数

        Returns:
            SD: [description]
        """
        self.data = [-data for data in self.data]
        return self

    def detrend(self) -> SD:
        """对data进行去趋势化操作"""
        detrended_data = signal.detrend(self.data).tolist()
        self.set_data(detrended_data)
        return self

    def remove_mean(self) -> SD:
        """将数据的baseline设为0附近"""
        mean_val = sum(self.data) / self.__data_length
        data_mean_value_as_baseline = [i-mean_val for i in self.data]
        self.set_data(data_mean_value_as_baseline)
        return self

    def average_smooth(self, window_size_sec: float) -> SD:
        """对data进行均值平滑操作"""
        window_size = int(window_size_sec * self.sample_rate)
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(np.array(self.data), box, mode='same')
        self.set_data(y_smooth.tolist())
        return self

    def savgol_filter_smooth(self, window_size_sec: float, poly_order: int = 2) -> SD:
        window_size = int(window_size_sec * self.sample_rate)
        window_size = window_size if (window_size & 1) == 1 else window_size+1
        res = signal.savgol_filter(
            self.to_nparray(), window_size, poly_order, mode='nearest')
        self.set_data(res.tolist())
        return self

    @staticmethod
    def __z_score_normalize(data: list) -> list:
        data_length = len(data)
        mean_val = sum(data) / data_length
        standard_deviation = math.sqrt(
            sum([(x-mean_val)**2 for x in data])/data_length)
        z_score_normalized_data = [
            (x-mean_val)/standard_deviation for x in data]
        return z_score_normalized_data

    def z_score_normalize(self) -> SD:
        """
        Z-score标准化
        z=x-mean/standard_deviation
        """
        z_score_normalized_data = self.__z_score_normalize(self.data)
        self.set_data(z_score_normalized_data)
        return self

    @staticmethod
    def __range_normalize(data: list, low_threshold: float = 0, high_threshold: float = 1) -> list:
        d_max = max(data)
        d_min = min(data)
        if d_max == d_min:
            return [0] * len(data)
        data = np.array(data)
        normalized_data = (data-d_min) / (d_max-d_min) * \
            (high_threshold-low_threshold) + low_threshold
        return normalized_data.tolist()

    def range_normalize(self, low_threshold: float = 0, high_threshold: float = 1) -> SD:
        """
        给定目标区间，进行归一化操作
        :param low_threshold: 目标数据最小值
        :param high_threshold: 目标数据最小值
        :return:
        """
        normalized_data = self.__range_normalize(
            self.data, low_threshold, high_threshold)
        self.set_data(normalized_data)
        return self

    def hampel_filter(self, filtsize: int = 6) -> SD:
        '''Detect outliers based on hampel filter

        Funcion that detects outliers based on a hampel filter.
        The filter takes datapoint and six surrounding samples.
        Detect outliers based on being more than 3std from window mean.
        '''
        output = self.to_nparray()
        onesided_filt = filtsize // 2
        for i in range(onesided_filt, self.__data_length - onesided_filt - 1):
            dataslice = output[i - onesided_filt: i + onesided_filt]
            median = np.median(dataslice)
            mad = np.median(np.abs(dataslice - median))
            if output[i] > median + (3 * mad):
                output[i] = median
        return self

    def spline_interpolate(self, kind: str = 'cubic') -> SD:
        """spiline interpolate self.data

        Args:
            kind (str, optional): [description]. Defaults to 'cubic'.

        Returns:
            SD: [description]
        """
        res = self.get_spline_interpolate_curve(self.data, kind)
        self.set_data(res)
        return self

    def difference_waveform(self) -> SD:
        """求差分曲线

        Returns:
            SD: [description]
        """
        self.set_data(self.get_difference_waveform(self.data))
        return self

    def get_sliced_data(self, start: int = None, end: int = None, step: int = None) -> SD:
        return get_SleepData_instance(self.sample_rate).load_data_from_list(self.data[start:end:step])

    @classmethod
    def get_range_normalized_data(cls, data: list, low_threshold: float = 0, high_threshold: float = 1) -> list:
        return cls.__range_normalize(data, low_threshold, high_threshold)

    @classmethod
    def get_z_score_normalized_data(cls, data: list):
        return cls.__z_score_normalize(data)

    @staticmethod
    def get_adaptive_window_normalized_data(data: list, window_size: int, a: float = 0.25, b: float = 0.125) -> list:
        """自适应加窗归一化
        [1] Peter Varady,“A novel method for thedetection of apnea and Hypopnea events in respiration signals”,
        IEEE Trans.Biomedical Engineering,Vol.49,No.09,September 2002

        Args:
            data (list): [description]
            a (float): [description]
            b (float): [description]
            window_size (int): [description]

        Returns:
            list: [description]
        """
        res_list = []
        segment_list = []
        for i in range(0, len(data), window_size):
            segment_list.append(data[i:i+window_size])
        if len(segment_list) == 0:
            return data
        if len(segment_list[-1]) != window_size:
            segment_list = segment_list[:-1]

        d_prev, f_prev = 2048, 800   # d0 is 2048, f0 is even, according to device
        for segment in segment_list:
            max_s = max(segment)
            min_s = min(segment)
            d = d_prev - a * (d_prev - (max_s+min_s) / 2)
            f = f_prev - b * (f_prev - max_s + min_s)
            d_prev = d
            f_prev = f
            res_list.extend([(s-d)/f for s in segment])
        return res_list

    @staticmethod
    def get_self_correlate_waveform(data_list: list) -> list:
        """计算self.data自相关曲线"""
        data_length = len(data_list)
        acf = np.correlate(data_list, data_list, mode='full')  # 自相关
        acf = acf[data_length-1:]
        acf = acf / acf[0]
        return acf.tolist()

    @staticmethod
    def get_difference_waveform(data_list: list) -> list:
        """求差分曲线"""
        data_length = len(data_list)
        gn1 = []
        for i in range(1, data_length - 1):
            gn1.append((data_list[i + 1] - data_list[i - 1]) / 2)

        gn2 = []
        for i in range(2, data_length - 2):
            gn2.append((2 * data_list[i + 1] + data_list[i + 2] -
                        2 * data_list[i - 1] - data_list[i - 2]) / 8)

        Gn = []
        for i in range(0, len(gn2)):
            Gn.append(gn1[i + 1] * gn1[i + 1] + gn2[i] * gn2[i])

        return Gn
    
    @staticmethod
    def detect_outliers_by_sigma(
        data_list: list, sample_rate: int, scales: float = 3.0,
        slide_window_secs: int = 5, window_size_secs: int = 5
    ) -> typing.List[int]:
        """3sigma-法则
        (μ-3sigma,μ+3sigma)区间内的概率为99.74。所以可以认为，当数据分布区间超过这个区间时，即可认为是异常数据。

        Args:
            data_list (list): 用于检测异常点的原始数据
            
        Returns:
            所有符合条件的异常点的下标
        """
        slide_stride = int(slide_window_secs * sample_rate)
        window_size = int(window_size_secs * sample_rate)

        all_outliers_index_list = []
        for start_index in tqdm(range(0, len(data_list), slide_stride)):
            end_index = start_index + window_size
            if end_index >= len(data_list):
                break
            temp_mean = float(np.mean(data_list[start_index:end_index]))
            temp_std = float(np.std(data_list[start_index:end_index]))
            high_threshold = temp_mean + scales * temp_std
            low_threshold = temp_mean - scales * temp_std
            index_list = list(range(start_index, end_index))
            part_outliers_index_list = list(filter(
                lambda x: data_list[x] > high_threshold or data_list[x] < low_threshold, index_list))
            all_outliers_index_list.extend(part_outliers_index_list)
        return all_outliers_index_list


    @staticmethod
    def detect_outliers_by_quantile(
        data_list: list, sample_rate: int, scales: float = 1.5,
        slide_window_secs: int = 5, window_size_secs: int = 5
    ) -> typing.List[int]:
        """分位数异常检测

            Args:
                data_list (list): 用于检测异常点的原始数据
            Returns:
                所有符合条件的异常点的下标
            """
        slide_stride = int(slide_window_secs * sample_rate)
        window_size = int(window_size_secs * sample_rate)

        first_quartile_index = int((window_size+1) * 0.25)
        third_quartile_index = int((window_size+1) * 0.75)

        all_outliers_index_list = []
        for start_index in tqdm(range(0, len(data_list), slide_stride)):
            end_index = start_index + window_size
            if end_index >= len(data_list):
                break
            temp_list = data_list[start_index:end_index]
            temp_list.sort()
            IQR = temp_list[third_quartile_index] - \
                temp_list[first_quartile_index]
            high_threshold = temp_list[third_quartile_index] + scales * IQR
            low_threshold = temp_list[first_quartile_index] - scales * IQR
            index_list = list(range(start_index, end_index))
            part_outliers_index_list = list(filter(
                lambda x: data_list[x] > high_threshold or data_list[x] < low_threshold, index_list))
            all_outliers_index_list.extend(part_outliers_index_list)
        return all_outliers_index_list

    def get_short_term_energy_waveform(self, window_size_sec: float = 0.3, padding: bool = True) -> list:
        """Calculate short term energy waveform according to paper:
        Robust heartbeat detection from in-home ballistocardiogram signals of older adults using a bed sensor

        Args:
            window_size_sec (float, optional): size of rolling window. Defaults to 0.3.

        Returns:
            list: short term energy waveform list
        """
        result_list = []
        for each_segment in self.get_sliding_windows(window_size_sec):
            result_list.append(sum([i**2 for i in each_segment]))
        result_list = self.__range_normalize(result_list, 0, 1)
        if padding is True:
            temp = [0 for i in range(int(window_size_sec*self.sample_rate)-1)]
            temp.extend(result_list)
            result_list = temp
        return result_list

    def get_processed_data_with_func(self, my_function, args: tuple = (), kwargs: dict = {}) -> typing.Any:
        """
        用自定义函数处理数据，自定义的函数至少接收一个list类型的数据
        自定义函数的参数传递可以使用args(元组)，kwargs(字典)传入
        """
        processed_data = my_function(self.get_data(), *args, **kwargs)
        return processed_data

    def get_data_values_from_indexes(self, indexes: list) -> list:
        """
        根据下标列表从self.data中取值，返回对应的值列表
        """
        return [self.data[index] for index in indexes if index < self.__data_length]

    def get_fft_frequency_spectrum(self) -> typing.Tuple[list, list]:
        """
        计算傅里叶变化的结果，用于频谱分析
        """
        # 去掉基线偏移，结果与matlab更接近
        mean_val = sum(self.data) / self.__data_length
        data = [i-mean_val for i in self.data]
        # sample spacing
        T = 1.0 / self.sample_rate
        yf = np.fft.fft(data)
        # xf = np.linspace(0.0, 1.0/(2.0*N*T), N//2)
        freqs = np.fft.fftfreq(self.__data_length, T)
        truey = [2.0 / self.__data_length *
                 np.abs(e) for e in yf[:self.__data_length // 2]]  # 归一化
        #x = np.linspace(0, len(y), len(y))
        return freqs[:len(freqs)//2], truey

    def get_lomb_scargle_periodogram(self) -> typing.Tuple[list, list]:
        """Caclulate Lomb-Scargle periodogram
        =====================================
        Lomb-Scargle periodogram has shown to better estimate the power spectral density (PSD) 
        of unevenly sampled data, such as HRV, than fast Fourier transform (FFT) based methods.

        Args:
            low_freq (float): lowest value of freq range
            high_freq (float): highest value of freq range
            is_normalize (bool): is compute normalized periodogram? Defaults to False.

        Returns:
            typing.Tuple[list, list]: freq range list, lomb-scargle pgram list
        """
        data_to_cal = self.to_nparray()
        freqs, res = LombScargle(
            np.array([i for i in range(self.__data_length)]), data_to_cal).autopower()
        return freqs[:len(freqs)//2], res[:len(freqs)//2]

    @staticmethod
    def __get_template_positions(
        acf: list, correlate_threshold: float = 0.3,
        min_template_length: float = 0.5, max_template_length: float = 2.0,
        max_template_count=6
    ) -> list:
        """
        @param correlate_threshold: 相关性阈值(衡量选模板的严格程度),推荐心率0.3,呼吸0.5
        @param min_template_length: 模板最小长度(秒*采样率)，推荐心率0.5s，呼吸2.0s
        @param max_template_length: 模板最大长度(秒*采样率)，推荐心率2.0s，呼吸5.0s
        @param max_template_count: 选出的模板数量上限(个),和时间片段长度有关，推荐30s6个
        """
        pattern_pos = []
        pos = 1
        for i in acf[1:]:
            if len(pattern_pos) > max_template_count:  # 模板数量上限
                break
            # 获取自相关后的位置
            if i > correlate_threshold and i > acf[pos-1] and i > acf[pos+1] and pos < max_template_length*(len(pattern_pos)+1):
                if len(pattern_pos) == 0:
                    pattern_pos.append(pos)
                else:
                    if pos - pattern_pos[-1] > min_template_length and \
                            pos - pattern_pos[-1] < max_template_length:
                        pattern_pos.append(pos)
            pos += 1
        return pattern_pos

    def calculate_signal_quality(
        self, correlate_threshold: float = 0.3, is_difference: bool = True,
        min_template_length_sec: float = 0.5, max_template_length_sec: float = 2.0,
        min_template_count: int = 4, max_template_count: int = 6,
    ) -> bool:
        """
        寻找模板，并通过模板数量判断信号质量
        @param correlate_threshold: 相关性阈值(衡量选模板的严格程度),推荐心率0.3,呼吸0.5
        @param is_difference: 是否对信号进行查分处理后再进行模板
        @param min_template_length_sec: 模板最小长度(秒)，推荐心率0.5，呼吸2.0
        @param max_template_length_sec: 模板最大长度(秒)，推荐心率2.0，呼吸5.0
        @param min_template_count: 选出的模板数量下限(个),和时间片段长度有关，推荐30s4个
        @param max_template_count: 选出的模板数量上限(个),和时间片段长度有关，推荐30s6个
        """
        min_template_length = int(self.sample_rate * min_template_length_sec)
        max_template_length = int(self.sample_rate * max_template_length_sec)

        data_backup = self.get_data()
        data_list = self.remove_mean().get_data()
        data_list = self.get_difference_waveform(data_list
        ) if is_difference is True else data_list    # 差分
        acf = self.get_self_correlate_waveform(data_list)
        self.set_data(data_backup)

        template_positions = self.__get_template_positions(
            acf, correlate_threshold,
            min_template_length, max_template_length, max_template_count
        )  # 获取模板的位置

        # 模板数量过少拒绝该窗口
        return False if len(template_positions) < min_template_count else True

    def get_hilbert_envelope(self) -> list:
        """求hilbert包络线"""
        analytical_signal = signal.hilbert(np.array(self.data))
        amplitude_envelope = np.abs(analytical_signal)
        # instantaneous_phase = np.unwrap(np.angle(analytical_signal))
        return amplitude_envelope.tolist()

    @staticmethod
    def get_spline_interpolate_curve(data_list: list, amount: int, kind='cubic') -> list:
        """获取经过样条插值之后的曲线
        @param amount: 插值之后的目标序列长度
        """
        x = [i for i in range(len(data_list))]
        res = interp1d(x, data_list, kind=kind)(
            np.linspace(0, len(data_list)-1, amount)).tolist()
        return res

    def get_peak_interval_sec_list(self, peak_list: typing.List[int]) -> typing.List[float]:
        # 计算峰值点之间的间隔序列，单位为秒
        peak_interval_sec_list = []
        for index in range(1, len(peak_list)):
            peak_interval_sec_list.append(
                (peak_list[index]-peak_list[index-1])/self.sample_rate)
        return peak_interval_sec_list

    def __detect_peaks(self, hrdata, rol_mean, ma_perc):
        # snippet from package heartpy.peakdetection, middle process of self.get_peak_list()
        rmean = np.array(rol_mean)

        #rol_mean = rmean + ((rmean / 100) * ma_perc)
        mn = np.mean(rmean / 100) * ma_perc
        rol_mean = rmean + mn

        peaksx = np.where((hrdata > rol_mean))[0]
        peaksy = hrdata[peaksx]
        peakedges = np.concatenate((np.array([0]),
                                    (np.where(np.diff(peaksx) > 1)[0]),
                                    np.array([len(peaksx)])))
        peaklist = []

        for i in range(0, len(peakedges)-1):
            try:
                y_values = peaksy[peakedges[i]:peakedges[i+1]].tolist()
                peaklist.append(
                    peaksx[peakedges[i] + y_values.index(max(y_values))])
            except:
                pass

        return peaklist

    def __remove_outliers_peaks(self, peak_list: typing.List[int], min_rr_interval_sec: float) -> typing.List[int]:
        '''find anomalous peaks.

        Funcion that checks peaks for outliers based on anomalous peak-peak distances and corrects
        by excluding them from further analysis.
        '''
        if len(peak_list) <= 1:
            return peak_list
        rr_arr = self.get_peak_interval_sec_list(peak_list)
        rr_arr = np.array(rr_arr)
        peak_list = np.array(peak_list)

        # define RR range as mean +/- 30%, with a minimum of 0.35sec
        mean_rr = np.mean(rr_arr)
        thirty_perc = 0.3 * mean_rr
        if thirty_perc <= 0.35:
            lower_threshold = mean_rr - 0.35
        else:
            lower_threshold = mean_rr - thirty_perc

        # 峰值点的间距至少要大于手工和算法结果两者的较大值
        lower_threshold = max(lower_threshold, min_rr_interval_sec)

        # identify peaks to exclude based on RR interval
        rem_idx = np.where((rr_arr <= lower_threshold))[0] + 1

        return [int(peak) for peak in peak_list if peak not in peak_list[rem_idx]]

    def __remove_edge_peaks(self, peak_list: list, edge_length_sec: float) -> typing.List[int]:
        """
        去除过于靠近片段两端的峰值点
        """
        if not peak_list:
            return []
        edge_length = int(self.sample_rate * edge_length_sec)
        new_peak_list = []
        for peak in peak_list:
            if peak < edge_length or peak > self.__data_length-edge_length:
                continue
            new_peak_list.append(peak)
        return new_peak_list

    def __get_local_maxium_peaks(self, peak_list: list, local_maxium_check_window_size_sec: float) -> typing.List[int]:
        """检测峰值点是否是周围点中的最大值(指定窗口中的极值点),是则保留，否则去除"""
        if not peak_list:
            return []
        new_peak_list = []
        local_maxium_check_window_size = int(
            self.sample_rate*local_maxium_check_window_size_sec)
        one_side_window_size = int(local_maxium_check_window_size // 2)
        for peak in peak_list:
            left_boundry = peak - \
                one_side_window_size if (peak-one_side_window_size) > 0 else 0
            right_boundry = peak+one_side_window_size if (
                peak+one_side_window_size) <= self.__data_length else self.__data_length
            if max(self.data[left_boundry:right_boundry]) != self.data[peak]:
                continue
            new_peak_list.append(peak)
        return new_peak_list

    def __get_verified_peak_list(
        self, peak_list: typing.List[int],
        min_peak_distance_sec: float = 0, edge_length_sec: float = 0.15,
        local_maxium_check_window_size_sec: float = 0.25
    ) -> typing.List[int]:
        """
        验证peak_list中的peak是否合理有效
        @param min_peak_distance_sec: 峰值点之间的最小间距(秒),推荐心率0.5，呼吸2.0
        @param edge_length_sec: 边缘的长度(秒)，推荐心率0.15,呼吸0.30
        @param local_maxium_check_window_size_sec: 检查峰值点是否是局部极大值的窗口大小(秒)[峰值点位于窗口中心]，推荐心率0.25,呼吸1.0
        """
        peak_list = self.__remove_edge_peaks(
            peak_list, edge_length_sec)   # 去除两端的peak
        peak_list = self.__get_local_maxium_peaks(
            peak_list, local_maxium_check_window_size_sec)  # 去除在指定窗口内非最大值的peak
        peak_list = self.__remove_outliers_peaks(
            peak_list, min_peak_distance_sec)   # 根据peak之间的距离去除一些距离过近的peak
        return peak_list

    @staticmethod
    def __get_sliding_windows(data_array: Union[np.array, list], window_size: int) -> np.array:
        '''segments data into windows

        Function to segment data into windows for rolling mean function.
        Function returns the data segemented into sections.

        For example:
        >>> data = [1,2,3,4,5,6]
        >>> sliding_windows = get_sliding_windows(data_array, 4)
        Now sliding_windows is [[1,2,3,4], [2,3,4,5], [3,4,5,6]]

        @return np.array[np.array, ...]
        '''
        data_array = np.array(data_array)
        shape = data_array.shape[:-1] + \
            (data_array.shape[-1] - window_size + 1, window_size)
        strides = data_array.strides + (data_array.strides[-1],)
        return np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)

    def get_sliding_windows(self, window_size_sec: float) -> typing.List[list]:
        '''segments data into windows

        Function to segment data into windows for rolling mean function.
        Function returns the data segemented into sections.

        For example:
        >>> data = [1,2,3,4,5,6]
        >>> sliding_windows = get_sliding_windows(data_array, 4)
        Now sliding_windows is [[1,2,3,4], [2,3,4,5], [3,4,5,6]]

        @return np.array[np.array, ...]
        '''
        data_array = self.to_nparray()
        window_size = int(window_size_sec * self.sample_rate)
        return self.__get_sliding_windows(data_array, window_size).tolist()

    @classmethod
    def complete_peak_list(cls, data_list: list, peak_list: list, max_peak_distance: int, min_peak_distance: int) -> typing.List[list]:
        """找到缺失的峰值点,并将其补充进去

        Args:
            data_list (list): 原始数据列表
            peak_list (list): 峰值点序列
            max_peak_distance (int): 峰值点之间的最大间距

        Returns:
            typing.List[list]: 补充后的峰值点列表
        """
        old_peak_list_length = len(peak_list)
        if old_peak_list_length == 0:
            return peak_list
        new_peak_list = [peak_list[0]]

        for i in range(1, old_peak_list_length):
            if int(peak_list[i] - peak_list[i-1]) > max_peak_distance:
                temp_list = data_list[peak_list[i-1]+min_peak_distance:peak_list[i]-min_peak_distance]
                local_maxium_index_list = cls.get_local_maxium_indexes(
                    temp_list)
                if not local_maxium_index_list:
                    new_peak_list.append(peak_list[i])
                    continue
                new_peak_list.append(
                    peak_list[i-1]+min_peak_distance+max(local_maxium_index_list, key=lambda x: temp_list[x]))
            else:
                new_peak_list.append(peak_list[i])

        return new_peak_list

    @staticmethod
    def remove_too_close_peaks(peak_list: list, min_peak_distance: int) -> typing.List[int]:
        """移除距离小于阈值的峰值点

        Args:
            peak_list (list): 峰值点序列
            min_peak_distance (int): 最小间距

        Returns:
            typing.List[int]: 新的峰值点序列
        """
        if len(peak_list) == 0:
            return []
        index = 1
        new_peak_list = [peak_list[0]]
        while index < len(peak_list):
            if peak_list[index]-new_peak_list[-1] > min_peak_distance:
                new_peak_list.append(peak_list[index])
            index += 1
        return new_peak_list

    @staticmethod
    def get_median_filtered_data(data_list: list, order: int = 3) -> list:
        """
        Low-pass filter. Replace each RRi value by the median of its ⌊N/2⌋
        neighbors. The first and the last ⌊N/2⌋ RRi values are not filtered

        Parameters
        ----------
        rri : array_like
            sequence containing the RRi series
        order : int, optional
            Strength of the filter. Number of adjacent RRi values used to calculate
            the median value to replace the current RRi. Defaults to 3.

        .. math::
            considering movinge average of order equal to 3:
                RRi[j] = np.median([RRi[j-2], RRi[j-1], RRi[j+1], RRi[j+2]])

        Returns
        -------
        results : RRi list
            instance of the RRi class containing the filtered RRi values
        """
        if len(data_list) < order-1:
            return []
        results = signal.medfilt(data_list, kernel_size=order).tolist()
        return results

    @classmethod
    def get_rolling_std(cls, data_list: list, sample_rate: int, window_size_secs: float, slide_window_secs: float, is_padding: bool = False) -> typing.List[float]:
        """计算移动平均标准差，用于衡量一个窗口的波动情况

        Args:
            data_list (list): 数据
            window_size_sec (float): 心率通常0.75，呼吸2.25
            is_padding (bool, optional): 是否填充数据到原来的长度. Defaults to False.

        Returns:
            typing.List[float]: 滑动标准差
        """
        slide_stride = int(slide_window_secs * sample_rate)
        window_size = int(window_size_secs * sample_rate)

        std_list = []
        for start_index in tqdm(range(0, len(data_list), slide_stride)):
            end_index = start_index + window_size
            if end_index >= len(data_list):
                break
            temp_std = float(np.std(data_list[start_index:end_index]))
            for _ in range(slide_stride-1):
                std_list.append(temp_std)
        std_list = np.array(std_list)

        if is_padding is True:
            # self.get_peak_list_by_move_avg()中调用时必须进行数据填充
            # need to fill 1/2 windowsize gap at the start and end
            n_missvals = int(abs(len(data_list) - len(std_list))/2)
            missvals_a = np.array([std_list[0]]*n_missvals)
            missvals_b = np.array([std_list[-1]]*n_missvals)

            std_list = np.concatenate((missvals_a, std_list, missvals_b))

        return std_list.tolist()

    def get_rolling_mean(self, window_size_sec: float, is_padding: bool = False, is_move_to_positive_baseline: bool = False) -> typing.List[float]:
        """
        计算移动平均值序列，传入窗口大小(秒)
        计算心率时, window_size_sec=0.75
        计算呼吸率时, window_size_sec=2.25

        @param is_padding: 是否将移动平均值序列填充到与数据序列一样长
        @param is_move_to_positive_baseline: 是否将原始数据序列移动到基线为正的位置
        """
        data_arr = self.to_nparray()

        if is_move_to_positive_baseline is True:
            # self.get_peak_list_by_move_avg()中调用时必须进行基线抬升
            # check that the data has positive baseline for the moving average algorithm to work
            bl_val = np.percentile(data_arr, 0.1)
            if bl_val < 0:
                data_arr = data_arr + abs(bl_val)

        rol_mean = np.mean(self.__get_sliding_windows(
            data_arr, int(window_size_sec*self.sample_rate)), axis=1)
        # remove a meanless 0 at last position of rol_mean list
        rol_mean = rol_mean[:-1]

        if is_padding is True:
            # self.get_peak_list_by_move_avg()中调用时必须进行数据填充
            # need to fill 1/2 windowsize gap at the start and end
            n_missvals = int(abs(len(data_arr) - len(rol_mean))/2)
            missvals_a = np.array([rol_mean[0]]*n_missvals)
            missvals_b = np.array([rol_mean[-1]]*n_missvals)

            rol_mean = np.concatenate((missvals_a, rol_mean, missvals_b))

            # only to catch length errors that sometimes unexplicably occur.
            # Generally not executed, excluded from testing and coverage
            if len(rol_mean) != self.__data_length:  # pragma: no cover
                lendiff = len(rol_mean) - self.__data_length
                if lendiff < 0:
                    rol_mean = np.append(rol_mean, 0)
        return rol_mean.tolist()

    def get_peak_list_by_move_avg(
        self, min_bpm: int, max_bpm: int, mean_precentage_list: list = 'auto',
        rolling_mean_window_size_sec: float = 0.75,
        min_peak_distance_sec: float = 0.5, edge_length_sec: float = 0.15,
        local_maxium_check_window_size_sec: float = 0.25
    ) -> typing.List[int]:
        # function refer to package heartpy function fit_peaks()
        # after get peak list, first judge if data surround the peak are smaller than peak, then detect outlier peak by peak-peak intervals
        # when detect heart curve peaks, use min_bpm=40 and max_bpm=180, rolling_mean_window_size_sec=0.75
        # when detect respiratory curve peaks, use min_bpm=12 and max_bpm=30, rolling_mean_window_size_sec=2.25
        """
        @param rolling_mean_window_size_sec: 滑动窗口的长度(秒),推荐心率0.75，呼吸2.0
        @param min_peak_distance_sec: 峰值点之间的最小间距(秒),推荐心率0.5，呼吸2.0
        @param edge_length_sec: 边缘的长度(秒)，推荐心率0.15,呼吸0.45
        @param local_maxium_check_window_size_sec: 检查峰值点是否是局部极大值的窗口大小(秒)[峰值点位于窗口中心]，推荐心率0.25,呼吸1.0
        """
        hrdata = self.to_nparray()
        rol_mean = np.array(self.get_rolling_mean(
            rolling_mean_window_size_sec, is_padding=True, is_move_to_positive_baseline=True))

        # (heartpy snippet)check that the data has positive baseline
        bl_val = np.percentile(hrdata, 0.1)
        if bl_val < 0:
            hrdata = hrdata + abs(bl_val)

        # moving average values to test (scale size)
        if mean_precentage_list == 'auto':
            ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50,
                            60, 70, 80, 90, 100, 110, 120, 150, 200, 300]
        else:
            ma_perc_list = mean_precentage_list

        rrsd = []
        valid_ma = []

        for ma_perc in ma_perc_list:
            peak_list = self.__detect_peaks(hrdata, rol_mean, ma_perc)
            bpm = (len(peak_list)/(len(hrdata)/self.sample_rate))*60
            rr_sec_list = self.get_peak_interval_sec_list(peak_list)
            _rrsd = np.std(np.array(rr_sec_list)) if len(
                rr_sec_list) > 0 else np.inf
            rrsd.append([_rrsd, bpm, ma_perc])

        for _rrsd, _bpm, _ma_perc in rrsd:
            if (_rrsd > 0.0001) and ((min_bpm <= _bpm <= max_bpm)):
                valid_ma.append([_rrsd, _ma_perc])

        if len(valid_ma) > 0:
            # best_mov_avg_scale = min(valid_ma, key=lambda t: t[0])[1] / 100
            peak_list = self.__detect_peaks(
                hrdata, rol_mean, min(valid_ma, key=lambda t: t[0])[1])
            peak_list = self.__get_verified_peak_list(
                peak_list, min_peak_distance_sec,
                edge_length_sec, local_maxium_check_window_size_sec
            )
            # TODO max_peak_distance check, utilize this to find missed peaks
            return [int(peak_list[i]) for i in range(len(peak_list))]
        else:
            return []

    @staticmethod
    def get_local_maxium_indexes(data_list: list) -> list:
        """calculate all local maxium indexes of data_list

        Args:
            data_list (list): [description]
        """
        peak_list = []
        for index in range(1, len(data_list)-1):
            if data_list[index-1] < data_list[index] and data_list[index] > data_list[index+1]:
                peak_list.append(index)
        return peak_list

    def get_peak_list_by_local_maxium(
        self, min_peak_distance_sec: float = 0, edge_length_sec: float = 0.15,
        local_maxium_check_window_size_sec: float = 0.25
    ) -> typing.List[int]:
        """
        根据极大值点求峰值，推荐用于信号质量好的呼吸信号，慎用于心率！！
        @param min_peak_distance_sec: 峰值点之间的最小间距(秒),推荐心率0.5，呼吸2.0
        @param edge_length_sec: 边缘的长度(秒)，推荐心率0.15,呼吸0.30
        @param local_maxium_check_window_size_sec: 检查峰值点是否是局部极大值的窗口大小(秒)[峰值点位于窗口中心]，推荐心率0.25,呼吸1.0
        """
        peak_list = self.get_local_maxium_indexes(self.data)

        peak_list = self.__get_verified_peak_list(
            peak_list, min_peak_distance_sec,
            edge_length_sec, local_maxium_check_window_size_sec
        )
        return peak_list

    @staticmethod
    def get_peak_list_by_scipy_find_peaks(
        data_list: list, distance: int, **kwargs
    ) -> typing.List[int]:
        """包装scipy.signal的find_peaks方法

        Args:
            data_list (list): 原始数据
            distance (int): 峰值点之间的最小水平距离
            其他需要传入find_peaks中的参数

        Returns:
            typing.List[int]: 峰值点下标列表
        """
        peak_list, _ = signal.find_peaks(data_list, distance=distance, **kwargs)
        return peak_list.tolist()

    @classmethod
    def adjust_peaks_position_to_local_maxium(cls, data_list: list, peak_list: list, left_search_points: int, right_search_points: int) -> list:
        """调整峰值点到局部极大值中的最大值的位置

        Args:
            data_list (list): 数值数据
            peak_list (list): 峰值点索引
            left_search_points (int): 向左搜索点的个数
            right_search_points (int): 向右搜索点的个数

        Returns:
            list: 调整后的峰值点列表
        """
        data_length = len(data_list)
        new_peak_list = []
        for peak in peak_list:
            left_side_index = peak - left_search_points if peak > left_search_points else 0
            right_side_index = peak + \
                right_search_points if (
                    peak+right_search_points) <= data_length else data_length
            temp_list = data_list[left_side_index:right_side_index]
            local_maxium_index_list = cls.get_local_maxium_indexes(temp_list)
            if not local_maxium_index_list:
                continue
            new_peak_list.append(
                left_side_index+max(local_maxium_index_list, key=lambda x: temp_list[x]))
        return new_peak_list

    @staticmethod
    def get_quartile_values(data_list: list) -> typing.Tuple[int, int, int]:
        """计算三个四等分点"""
        sorted_data_list = sorted(data_list)
        length = len(sorted_data_list)
        first_quartile_index = int((length+1) * 0.25)
        second_quartile_index = int((length+1) * 0.5)
        third_quartile_index = int((length+1) * 0.75)
        return sorted_data_list[first_quartile_index], sorted_data_list[second_quartile_index], sorted_data_list[third_quartile_index]

    @staticmethod
    def get_detrended_data(data_list: list) -> list:
        """对data进行去趋势化操作"""
        detrended_data = signal.detrend(data_list).tolist()
        return detrended_data

    @staticmethod
    def get_scaled_data(data_list: list, scale: float) -> list:
        """输出放大指定倍数后的序列

        Args:
            data_list (list): 数据
            scale (float): 放大倍数

        Returns:
            list: [description]
        """
        return [i*scale for i in data_list]

    @staticmethod
    def get_rri_list(peak_list: list) -> list:
        """根据峰值点序列计算rri

        Args:
            peak_list (list): [description]

        Returns:
            list: [description]
        """
        return [peak_list[i]-peak_list[i-1] for i in range(1, len(peak_list))]

    def split_data_to_segments(self, segment_length: int, is_abandon_last: bool = True, to_list: bool = False) -> typing.List[Union[list, SD]]:
        """按指定长度切分数据,is_abandon_last是否丢弃最后一个不符合长度的片段"""
        res_list = []
        for i in range(0, self.__data_length, segment_length):
            res_list.append(self.data[i:i+segment_length])
        if is_abandon_last is True and len(res_list[-1]) != segment_length:
            res_list = res_list[:-1]
        if to_list is False:
            for i in range(len(res_list)):
                res_list[i] = get_SleepData_instance(
                    self.sample_rate).load_data_from_list(res_list[i])
        return res_list


def get_SleepData_instance(sample_rate: float = None) -> SleepData:
    return SleepData(sample_rate=sample_rate)


def iir_high_pass_filter(
    numlist: list, sample_rate: int, cutoff: float,
    filter_order: int = 4, filter_type: str = 'butter',
) -> np.array:
    """
    高通滤波器
    numlist:      data need to be filter
    cutoff:       cutoff rate(Hz)
    sample_rate:  sample rate of data(Hz)
    filter_order: order of filter you choose
    filter_type:  'butter' or 'bessel'
    @return:      data after high pass filter
    """
    fs = sample_rate
    fc = cutoff  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    if filter_type == 'butter':
        b, a = signal.butter(filter_order, w, 'high')
    elif filter_type == 'bessel':
        b, a = signal.bessel(filter_order, w, 'high')
    else:
        raise TypeError("Filter type is not supported!")
    output = signal.filtfilt(b, a, numlist)
    return output


def iir_low_pass_filter(
    numlist: list, sample_rate: int, cutoff: float,
    filter_order=4, filter_type='butter',
) -> np.array:
    """
    低通滤波器
    numlist:      data need to be filter
    cutoff:       cutoff rate(Hz)
    sample_rate:  sample rate of data(Hz)
    filter_order: order of filter you choose
    filter_type:  'butter' or 'bessel'
    @return:      data after high pass filter
    """
    fs = sample_rate
    fc = cutoff  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    if filter_type == 'butter':
        b, a = signal.butter(filter_order, w, 'low')
    elif filter_type == 'bessel':
        b, a = signal.bessel(filter_order, w, 'low')
    else:
        raise TypeError("Filter type is not supported!")
    output = signal.filtfilt(b, a, numlist)
    return output


def read_sleep_data_column(filename: str, column: int, to_which_type: type = int):
    if column < 0:
        raise ValueError("column must greater than or equal to zero")
    result = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split()
            if len(temp) <= column:
                continue
            result.append(to_which_type(temp[column]))
    return result
