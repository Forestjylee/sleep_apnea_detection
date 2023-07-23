import os
import glob
import typing

import pandas as pd
import plotly.express as px
from plotly_resampler import register_plotly_resampler
from dash import Dash, Input, Output, callback, dcc, html

from config import settings
from sleep_data_obj import SleepData
from utils import (
    check_path_exist,
    get_sleep_apnea_label_list_according_to_data_length,
    read_pickle,
    plot_anything,
    transform_label_data_to_uniform_format,
    SLEEP_APNEA_EVENT_MAPPER,
)


def prepare_data_for_dash_display(
    sensor_data_dict: typing.Dict[str, list], sleep_apnea_label_list: list
) -> pd.DataFrame:
    data_to_display = []
    for idx in range(len(sleep_apnea_label_list)):
        tmp_dict = {
            "sleep_apnea_label": sleep_apnea_label_list[idx],
        }
        for key in sensor_data_dict.keys():
            tmp_dict[key] = sensor_data_dict[key][idx]
        data_to_display.append(tmp_dict)

    df_data_to_display = pd.DataFrame().from_dict(data_to_display)
    return df_data_to_display


def _check_sensor_data_dict_data_length(
    sensor_data_dict: typing.Dict[str, list]
) -> None:
    data_length = len(next(iter(sensor_data_dict.values())))
    if sensor_data_dict:
        for key in sensor_data_dict.keys():
            assert (
                len(sensor_data_dict[key]) == data_length
            ), f"Sensor data {key} length is not equal"
    else:
        ...


def get_corresponding_sleep_apnea_label_list(
    record_name: str,
    sample_rate: int,
    data_length: int,
    sleep_apnea_label_folder: str,
    is_map_to_text_label: bool = True,
) -> typing.List[int]:
    sleep_apnea_label_path = os.path.join(
        sleep_apnea_label_folder, f"{record_name}_sa_events.pkl"
    )
    raw_sleep_apnea_label_data = read_pickle(sleep_apnea_label_path)
    uniform_format_sleep_apnea_label_data = transform_label_data_to_uniform_format(
        raw_sleep_apnea_label_data, sample_rate
    )
    sleep_apnea_label_list = get_sleep_apnea_label_list_according_to_data_length(
        uniform_format_sleep_apnea_label_data, data_length
    )

    if is_map_to_text_label is True:
        sleep_apnea_label_list = list(
            map(lambda x: SLEEP_APNEA_EVENT_MAPPER[x], sleep_apnea_label_list)
        )

    return sleep_apnea_label_list


def plot_sensor_and_label_data_use_dash(
    sensor_names: list,
    record_name: str,
    sample_rate: int,
    raw_data_folder: str,
    sleep_apnea_label_folder: str,
    title: str = "Sensor data and label data",
) -> None:
    sensor_data_dict = {}
    for sensor_name in sensor_names:
        raw_data_path = os.path.join(
            raw_data_folder, f"{record_name}_{sensor_name}.pkl"
        )
        check_path_exist(raw_data_path)
        raw_data_dict = read_pickle(raw_data_path)
        raw_data_obj = (
            SleepData(raw_data_dict["sample_rate"])
            .load_data_from_list(raw_data_dict["data"])
            .resample(sample_rate)
            .range_normalize(0, 3)
        )
        raw_data_obj += 3
        sensor_data_dict[sensor_name] = raw_data_obj.get_data()

    _check_sensor_data_dict_data_length(sensor_data_dict)

    sleep_apnea_label_list = get_corresponding_sleep_apnea_label_list(
        record_name=record_name,
        sample_rate=sample_rate,
        data_length=len(next(iter(sensor_data_dict.values()))),
        sleep_apnea_label_folder=sleep_apnea_label_folder,
    )

    df_data_to_display = prepare_data_for_dash_display(
        sensor_data_dict, sleep_apnea_label_list
    )

    register_plotly_resampler()
    fig = px.line(
        df_data_to_display,
        x=df_data_to_display.index,
        y=df_data_to_display.columns,
        title=title,
    )
    fig.show()


def plot_sensor_and_label_data_use_matplotlib(
    sensor_names: list,
    record_name: str,
    sample_rate: int,
    raw_data_folder: str,
    sleep_apnea_label_folder: str,
    title: str = "Sensor data and label data",
) -> None:
    sensor_data_dict = {}
    for sensor_name in sensor_names:
        raw_data_path = os.path.join(
            raw_data_folder, f"{record_name}_{sensor_name}.pkl"
        )
        check_path_exist(raw_data_path)
        raw_data_dict = read_pickle(raw_data_path)
        raw_data_obj = (
            SleepData(raw_data_dict["sample_rate"])
            .load_data_from_list(raw_data_dict["data"])
            .resample(sample_rate)
            .range_normalize()
        )
        sensor_data_dict[sensor_name] = raw_data_obj.get_data()

    _check_sensor_data_dict_data_length(sensor_data_dict)

    sleep_apnea_label_list = get_corresponding_sleep_apnea_label_list(
        record_name=record_name,
        sample_rate=sample_rate,
        data_length=len(next(iter(sensor_data_dict.values()))),
        sleep_apnea_label_folder=sleep_apnea_label_folder,
    )

    data_to_plot = [sensor_data_dict[sensor_name] for sensor_name in sensor_names]
    data_to_plot.append(sleep_apnea_label_list)
    plot_anything(
        data_to_plot, y_labels=[*sensor_names, "sleep apnea label"], title=title
    )


def plot_sensor_data_dict_and_label_data_use_matplotlib(
    record_name: str,
    sensor_data_dict: typing.Dict[str, list],
    sleep_apnea_label_folder: str,
    title: str = "Sensor data and label data",
) -> None:
    _check_sensor_data_dict_data_length(sensor_data_dict)

    sleep_apnea_label_list = get_corresponding_sleep_apnea_label_list(
        record_name=record_name,
        sample_rate=sample_rate,
        data_length=len(next(iter(sensor_data_dict.values()))),
        sleep_apnea_label_folder=sleep_apnea_label_folder,
    )

    data_to_plot = [sensor_data_dict[sensor_name] for sensor_name in sensor_names]
    data_to_plot.append(sleep_apnea_label_list)
    plot_anything(
        data_to_plot, y_labels=[*sensor_names, "sleep apnea label"], title=title
    )


if __name__ == "__main__":
    raw_data_folder = settings.shhs1_raw_data_path
    sleep_apnea_label_folder = settings.shhs1_sleep_apnea_label_path

    sensor_names = ["ABDO", "THOR", "NEW"]
    record_name = "shhs1-200001"
    sample_rate = settings.resample_rate

    ## Matplotlib(recommanded)
    plot_sensor_and_label_data_use_matplotlib(
        sensor_names=sensor_names,
        record_name=record_name,
        sample_rate=sample_rate,
        raw_data_folder=raw_data_folder,
        sleep_apnea_label_folder=sleep_apnea_label_folder,
        title=f"{record_name}'s sensor data and label data",
    )

    ## Dash
    ## Native dash is hard to display too many data points(>100000)
    ## So we use plotly_resampler to downsample the data points to display
    ## But details of downsampled data are not accurate
    # plot_sensor_and_label_data_use_dash(
    #     sensor_names=sensor_names,
    #     record_name=record_name,
    #     sample_rate=sample_rate,
    #     raw_data_folder=raw_data_folder,
    #     sleep_apnea_label_folder=sleep_apnea_label_folder,
    #     title=f"{record_name}'s sensor data and label data"
    # )
