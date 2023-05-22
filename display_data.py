import os
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


def plot_sensor_and_label_data_to_dash(
    df_data_to_display: pd.DataFrame, title: str = "Sensor and Label Data"
):
    register_plotly_resampler()
    fig = px.line(
        df_data_to_display,
        x=df_data_to_display.index,
        y=df_data_to_display.columns,
        title=title,
    )
    fig.show()


if __name__ == "__main__":
    raw_data_folder = settings.shhs1_raw_data_path
    sleep_apnea_label_folder = settings.shhs1_sleep_apnea_label_path

    sensor_names = ["ABDO", "THOR", "NEW"]
    record_name = "shhs1-200001"
    sample_rate = settings.sample_rate

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

    sleep_apnea_label_path = os.path.join(
        sleep_apnea_label_folder, f"{record_name}_sa_events.pkl"
    )
    raw_sleep_apnea_label_data = read_pickle(sleep_apnea_label_path)
    uniform_format_sleep_apnea_label_data = transform_label_data_to_uniform_format(
        raw_sleep_apnea_label_data, sample_rate
    )
    sleep_apnea_label_list = get_sleep_apnea_label_list_according_to_data_length(
        uniform_format_sleep_apnea_label_data, len(sensor_data_dict[sensor_names[0]])
    )

    # Matplotlib(recommanded)
    data_to_plot = [sensor_data_dict[sensor_name] for sensor_name in sensor_names]
    data_to_plot.append(sleep_apnea_label_list)
    plot_anything(
        data_to_plot,
        y_labels=[*sensor_names, "sleep apnea label"]
    )

    # Dash
    # Native dash is hard to display too many data points(>100000)
    # So we use plotly_resampler to downsample the data points to display
    # But details of downsampled data are not accurate
    df_data_to_display = prepare_data_for_dash_display(
        sensor_data_dict, sleep_apnea_label_list
    )
    plot_sensor_and_label_data_to_dash(
        df_data_to_display, title=f"{record_name}'s Sensor Data and Sleep Apnea Labels"
    )
