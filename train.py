import glob
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from alive_progress import alive_bar
from tqdm import tqdm

keras = tf.keras
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from keras.utils import to_categorical
from sklearn.utils import class_weight as class_weight_obj

from config import settings
from models.transformer import build_transformer_model
from sleep_data_obj import SleepData
from utils import label_interceptor, read_pickle, save_as_pickle

# config GPU
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

l2 = keras.regularizers.l2


def get_train_and_validate_data(sensor_name: str):
    train_record_names, validation_record_names = [], []
    X_train, y_train = [], []
    X_validation, y_validation = [], []

    train_data_folder = Path(settings.shhs1.train_data_path)
    with alive_bar(title="Loading train data") as pbar:
        for each_train_data_path in train_data_folder.glob(f"*_{sensor_name}_X_data.pkl"):
            train_data_name = each_train_data_path.name
            record_name = train_data_name.split('_')[0]
            X_train.extend(read_pickle(Path(train_data_folder, f"{record_name}_{sensor_name}_X_data.pkl").absolute()))
            y_train.extend(read_pickle(Path(train_data_folder, f"{record_name}_{sensor_name}_y_data.pkl").absolute()))
            pbar()
    
    validation_data_folder = Path(settings.shhs1.validation_data_path)
    with alive_bar(title="Loading validation data") as pbar:
        for each_validation_data_path in validation_data_folder.glob(f"*_{sensor_name}_X_data.pkl"):
            validation_data_name = each_validation_data_path.name
            record_name = validation_data_name.split('_')[0]
            X_validation.extend(read_pickle(Path(validation_data_folder, f"{record_name}_{sensor_name}_X_data.pkl").absolute()))
            y_validation.extend(read_pickle(Path(validation_data_folder, f"{record_name}_{sensor_name}_y_data.pkl").absolute()))
            pbar()
    
    y_train = label_interceptor(y_train)
    y_validation = label_interceptor(y_validation)
    
    # X_train, y_train = SMOTE(sampling_strategy="minority").fit_resample(X_train, y_train)
    
    return np.array(X_train), np.array(X_validation), np.array(y_train), np.array(y_validation)


def train():
    model_filepath = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{settings.train.model_name}.h5"
    segment_length = int(settings.preprocess.window_length_secs * settings.preprocess.resample_rate)
    
    model = build_transformer_model((segment_length, 1), settings.train.class_num, lr=settings.train.learning_rate)
    
    print ('\nData Loaded. Compiling...\n')
    print('Loading data... ')
    ## SHHS-1
    X_train, X_validation, y_train, y_validation= get_train_and_validate_data("ABDO")  # ABDO | THOR | NEW

    ## MESA
    # X_train, X_validation, y_train, y_validation= get_train_and_validate_data("ABDO")  # Abdo | Thor | Flow

    # feature engneering
    # print(X_train)
    # print(X_train.shape)
    # exit(1)
    # X_train = X_train.transpose((0, 2, 1))
    # X_validation = X_validation.transpose((0, 2, 1))

    class_w = class_weight_obj.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    print(class_w)
    
    y_train = to_categorical(y_train, settings.train.class_num)
    y_validation = to_categorical(y_validation, settings.train.class_num)

    try:
        print("Training...")
        
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)
        select_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_filepath, monitor='val_loss', mode='min',
            save_best_only=True, verbose =1, period=1, save_weights_only=False
        )

        class_weight = {i : class_w[i] for i in range(len(class_w))}
        # class_weight = {
        #     0: 1,
        #     1: 4,
        # }
        history = model.fit(X_train, y_train, epochs=settings.train.epochs, batch_size=settings.train.batch_size, 
                            class_weight=class_weight,
                            validation_data=(X_validation, y_validation),
                            # validation_split=0.1,
                            callbacks=[earlystop_callback, select_best_checkpoint])
        pd.DataFrame(history.history).to_csv('training_log.csv', index=False)

        print(f"Save model to {model_filepath} success")

    except KeyboardInterrupt:
        print("prediction exception")
        return model


def plot_history():
    # 读取保存后的训练日志文件
    df = pd.read_csv('training_log.csv')

    # 画训练曲线
    def plot_learning_curves(df):
        df.plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    plot_learning_curves(df)


if __name__ == '__main__':
    train()
    plot_history()
