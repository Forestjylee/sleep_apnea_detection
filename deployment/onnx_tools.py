import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,
                             classification_report, auc, roc_curve, confusion_matrix, precision_recall_curve)

from mlxtend.evaluate import confusion_matrix as ploted_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sleep_data_obj import SleepData
from model_backup.abdo_perposed_per_segment.transformer_encoder import build_transformer_encode_model

import tf2onnx
import onnxruntime as rt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

CLASSES_NUM = 2

trained_model_folder = r"E:\developer\SleepProject\sleep_apnea_lab\shhs_experiment\model_backup\abdo_perposed_per_segment"
train_dataset_folder = r"E:\developer\SleepProject\sleep_apnea_lab\shhs_experiment\train_data"
test_dataset_folder = r"E:\developer\SleepProject\sleep_apnea_lab\shhs_experiment\test_data"


def read_pickle(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def label_interceptor(label_list: list):
    new_label_list = []
    normal_amount = 0
    for label in label_list:
        if label == 0:
            normal_amount += 1
            new_label_list.append(0)
        else:
            new_label_list.append(1)
        # else:
        #     new_label_list.append(label)
    print(
        f"Normal segments: {normal_amount}, abnormal segments: {len(label_list)-normal_amount}")
    return new_label_list


def get_all_data(sensor_name: str):
    record_names = []
    total_X, total_y = [], []

    for name in glob.glob(f"train_data/*_{sensor_name}_X_data.pkl")[1500:1501]:
        record_name = name.split('\\')[1].split('_')[0]
        if record_name not in record_names:
            record_names.append(record_name)

    train_record_names = record_names  # if index not in
    for record_name in tqdm(train_record_names, desc="Reading train data: "):
        total_X.extend(read_pickle(os.path.join(
            train_dataset_folder, f"{record_name}_{sensor_name}_X_data.pkl")))
        total_y.extend(read_pickle(os.path.join(
            train_dataset_folder, f"{record_name}_{sensor_name}_y_data.pkl")))

    total_y = label_interceptor(total_y)
    return total_X, total_y


def log_print_inference(y_test, y_pred, label_value, target_names, epochs=0, tensor_board_path='', file_title=""
                        , avg_method='macro'):
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
    # y_test = np.argmax(y_test, axis=-1)
    # y_pred = np.argmax(y_pred, axis=-1)
    # if len(y_pred.shape) > 2:
    #     y_pred = np.reshape(y_pred, -1)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(y_test, y_pred, average=avg_method)
    print('Avg Precision: %f' % precision)
    recall = recall_score(y_test, y_pred, average=avg_method)
    print('Avg Recall: %f' % recall)
    f1_result = f1_score(y_test, y_pred, average='macro')
    print('Macro F1 score: %f' % f1_result)
    af1_result = f1_score(y_test, y_pred, average='micro')
    print('Micro F1 score: %f' % af1_result)
    # auc_result = auc(y_test, y_pred)
    # print('AUC score: %f' % auc_result)
    kappa = cohen_kappa_score(y_test, y_pred)
    print('cohen kappa: %f' % kappa)
    report = classification_report(y_test, y_pred, labels=label_value, target_names=target_names)
    print("Classification report: \n")
    print(report)
    to_json = {'epoch_num': epochs, 'accuracy': accuracy, 'precision_weighted': precision, 'recall': recall,
               'f1_result': f1_result}
    result = pd.DataFrame.from_dict(to_json, orient='index')
    result.to_csv(os.path.join(tensor_board_path, file_title + "metrics_summary.csv"), index=False)
    with open(os.path.join(tensor_board_path, file_title + "classification_report.txt"), "w") as text_file:
        text_file.write(report)
        

def plot_matrix(y_target, y_pred, class_names: list, title_index: int=0):
    cm = ploted_confusion_matrix(
        y_target=y_target, 	
        y_predicted=y_pred, 	
        binary=False
    )	
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=class_names, figsize=(7, 7))	
    plt.savefig(f"{title_index}_confusion_matrix.svg")
    
    
def get_balanced_data(X: np.ndarray, y: np.ndarray):
    rus = RandomUnderSampler(random_state=42)
    # tl = TomekLinks(n_jobs=8)
    print('Original dataset shape %s' % Counter(y))
    # X, y = tl.fit_resample(X, y)
    # print('After TomekLink dataset shape %s' % Counter(y))
    X, y = rus.fit_resample(X, y)
    print('After random under sample dataset shape %s' % Counter(y))
    return X, y
    
    
def model_evaluate(model, X_test, y_test, title_index: int=0):
    y_test = label_interceptor(y_test)
    # y_test = to_categorical(y_test, 2)
    
    X_test = np.array(X_test)
    # X_test2 = np.array(X_test2)
    raw_y_pred = model.predict([X_test], verbose=1)
    y_pred = np.argmax(raw_y_pred, axis=-1)
    
    print('-'*50)
    label_values = [0, 1]
    class_names = ["normal", "apnea"]
    
    plot_matrix(y_test, y_pred, class_names=class_names, title_index=title_index)
    C = confusion_matrix(y_test, y_pred, labels=label_values)
    FP = C.sum(axis=0) - np.diag(C)  
    FN = C.sum(axis=1) - np.diag(C)
    TP = np.diag(C)
    TN = C.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    print(f"Sensitivity: {TPR}, Specificity: {TNR}, Precision: {PPV}, NPV: {NPV}, Accuracy: {ACC}")
    log_print_inference(y_test, y_pred, label_values, class_names, file_title="test_")
    return raw_y_pred


if __name__ == '__main__':
    # Traditional tensorflow predict
    sensor_name = "ABDO"      # ABDO | THOR | NEW
    total_X, total_y = get_all_data(sensor_name)
    total_X = np.array(total_X)
    total_X = total_X.reshape(total_X.shape[0], -1)
    total_y = np.array(total_y)
    
    # total_X, total_y = get_balanced_data(total_X, total_y)    # under sampling
    total_X = total_X.reshape(total_X.shape[0], 300, 1)
    
    transformer_model = build_transformer_encode_model((300, 1), CLASSES_NUM)
    transformer_model.load_weights(os.path.join(trained_model_folder, f"4_fold_model.h5"))
    st = time.time()
    raw_y_pred = model_evaluate(transformer_model, total_X, total_y)
    print(f"TensorFlow predict cost: {time.time() - st}")
    
    # Convert trained tensorflow model to onnx format (3~4x faster than TensorFlow GPU calculation)
    # onnx runtime
    spec = (tf.TensorSpec((None, 300, 1), tf.float32, name="input"),)
    output_path = "abdo_30s_CL_TransNet.onnx"
    # model_proto, _ = tf2onnx.convert.from_keras(transformer_model, input_signature=spec, output_path=output_path)
    # output_names = [n.name for n in model_proto.graph.output]
    
    providers = ['CPUExecutionProvider']   # 'CPUExecutionProvider' or 'CUDAExecutionProvider'
    m = rt.InferenceSession(output_path, providers=providers)
    st = time.time()
    onnx_pred = m.run(None, {"input": total_X})
    print(f"ONNX predict cost: {time.time() - st}")

    print('ONNX Predicted:', onnx_pred[0])

    # make sure ONNX and keras have the same results
    np.testing.assert_allclose(raw_y_pred, onnx_pred[0], rtol=1e-4)
