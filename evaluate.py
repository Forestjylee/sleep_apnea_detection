import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras
from collections import Counter

import seaborn as sns
from alive_progress import alive_bar
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
from keras.utils import to_categorical
from mlxtend.evaluate import confusion_matrix as ploted_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from config import settings
from models.transformer import build_transformer_model
from sleep_data_obj import SleepData
from utils import label_interceptor, path_join_trained_models_folder, read_pickle

sns.set(font_scale=1.1)
sns.set_style("ticks")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_test_data(sensor_name: str, test_data_folder: str):
    X_test, y_test = [], []

    with alive_bar(title="Loading test data") as pbar:
        for each_test_data_path in list(
            test_data_folder.glob(f"*_{sensor_name}_X_data.pkl")
        )[:50]:
            test_data_name = each_test_data_path.name
            record_name = test_data_name.split("_")[0]
            X_test.extend(
                read_pickle(
                    Path(
                        test_data_folder, f"{record_name}_{sensor_name}_X_data.pkl"
                    ).absolute()
                )
            )
            y_test.extend(
                read_pickle(
                    Path(
                        test_data_folder, f"{record_name}_{sensor_name}_y_data.pkl"
                    ).absolute()
                )
            )
            pbar()

    y_test = label_interceptor(y_test)

    return np.array(X_test), np.array(y_test)


def log_print_inference(
    y_test,
    y_pred,
    label_value,
    target_names,
    avg_method="macro",
):
    """
    Log inference results to tensor board path, we can track each experiment prediction result include accuracy, recall,
    precision, F1 score, F1 report, confusion matrix and confusion matrix in picture format.
    y_test is predict result
    y_pred is real label
    AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
    0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
    AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
    AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测
    """
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %f" % accuracy)
    precision = precision_score(y_test, y_pred, average=avg_method)
    print("Avg Precision: %f" % precision)
    recall = recall_score(y_test, y_pred, average=avg_method)
    print("Avg Recall: %f" % recall)
    f1_result = f1_score(y_test, y_pred, average="macro")
    print("Macro F1 score: %f" % f1_result)
    af1_result = f1_score(y_test, y_pred, average="micro")
    print("Micro F1 score: %f" % af1_result)
    # auc_result = auc(y_test, y_pred)
    # print("AUC score: %f" % auc_result)
    kappa = cohen_kappa_score(y_test, y_pred)
    print("cohen kappa: %f" % kappa)
    report = classification_report(
        y_test, y_pred, labels=label_value, target_names=target_names
    )
    print("Classification report: \n")
    print(report)


def plot_matrix(y_target, y_pred, class_names: list, title_index: int = 0):
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
    cm = ploted_confusion_matrix(y_target=y_target, y_predicted=y_pred, binary=False)

    fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=class_names, figsize=None)
    ax.set_xlabel("预测类型")
    ax.set_ylabel("真实类型")
    ax.tick_params(axis="x", labelsize=18, rotation=0)
    ax.tick_params(axis="y", labelsize=18, rotation=0)
    plt.savefig(f"{title_index}_confusion_matrix.svg", bbox_inches="tight")
    plt.show()


def get_balanced_data(X: np.ndarray, y: np.ndarray):
    rus = RandomUnderSampler(random_state=42)
    # tl = TomekLinks(n_jobs=8)
    print("Original dataset shape %s" % Counter(y))
    # X, y = tl.fit_resample(X, y)
    # print('After TomekLink dataset shape %s' % Counter(y))
    X, y = rus.fit_resample(X, y)
    print("After random under sample dataset shape %s" % Counter(y))
    return X, y


def model_evaluate(model, X_test, y_test, title_index: int = 0):
    class_num = settings.train.class_num

    y_test = label_interceptor(y_test)
    X_test = np.array(X_test)

    y_pred = model.predict([X_test], verbose=1)
    y_pred = np.argmax(y_pred, axis=-1)

    print("#" * 50)

    # label_values = [0, 1, 2, 3, 4]
    # class_names = ["normal", "hyponea", "obstructive", "central", "mix"]
    label_values = [0, 1]
    class_names = ["normal", "apnea/hyponea"]
    # label_values = [0, 1, 2, 3]
    # class_names = ["normal", "hyponea", "obstructive", "central"]

    plot_matrix(y_test, y_pred, class_names=class_names, title_index=title_index)
    C = confusion_matrix(y_test, y_pred, labels=label_values)
    FP = C.sum(axis=0) - np.diag(C)
    FN = C.sum(axis=1) - np.diag(C)
    TP = np.diag(C)
    TN = C.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print(
        f"Sensitivity: {TPR}, Specificity: {TNR}, Precision: {PPV}, NPV: {NPV}, Accuracy: {ACC}"
    )
    print(
        f"Sensitivity: {np.sum(TPR)/class_num}\n, Specificity: {np.sum(TNR)/class_num}\n, Precision: {np.sum(PPV)/class_num}\n, NPV: {np.sum(NPV)/class_num}\n, Accuracy: {np.sum(ACC)/class_num}"
    )
    log_print_inference(y_test, y_pred, label_values, class_names)


if __name__ == "__main__":
    model_filepath = r"20230818_142357_cnn_transformer.h5"
    model_filepath = path_join_trained_models_folder(model_filepath)

    test_data_folder = Path(settings.shhs1.test_data_path)

    segment_length = int(
        settings.preprocess.window_length_secs * settings.preprocess.resample_rate
    )

    sensor_name = "ABDO"  # ABDO | THOR | NEW
    total_X, total_y = get_test_data(sensor_name, test_data_folder)

    ## Unsampling test data
    # total_X = total_X.reshape(total_X.shape[0], -1)
    # total_X, total_y = get_balanced_data(total_X, total_y)    # under sampling
    # total_X = total_X.reshape(total_X.shape[0], segment_length, 1)

    transformer_model = build_transformer_model(
        total_X.shape[1:], settings.train.class_num
    )
    transformer_model.load_weights(model_filepath)
    model_evaluate(transformer_model, total_X, total_y)
