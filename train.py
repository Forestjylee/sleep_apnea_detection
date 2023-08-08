import os
import time
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
keras = tf.keras
from keras.utils import to_categorical
from sklearn.utils import class_weight as class_weight_obj
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sleep_data_obj import SleepData
from models.transformer import build_transformer_model

train_dataset_folder = r"E:\developer\PythonProject\Sleep\sleep_apnea_lab\shhs_experiment\train_data"

# config GPU
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

CLASSES_NUM = 2    # 分类数

l2 = keras.regularizers.l2

model_filepath = "latest_model.h5"


def read_pickle(filename: str):
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


def save_as_pickle(data, filename: str):
    with open(filename, 'wb') as fw:
        pickle.dump(data, fw)


def label_interceptor(label_list: list):
    new_label_list = []
    for label in label_list:
        if label ==0:
            new_label_list.append(0)
        # elif label == 1:
        #     new_label_list.append(2)
        else:
            new_label_list.append(1)
    return new_label_list


def get_train_and_validate_data(sensor_name: str):
    record_names = []
    X_train, y_train = [], []
    X_validation, y_validation = [], []

    for name in glob.glob(f"train_data/*_{sensor_name}_X_data.pkl"):
        record_name = name.split('\\')[1].split('_')[0]
        if record_name not in record_names:
            record_names.append(record_name)
        
    validation_record_names = record_names[-300:]
    train_record_names = record_names[:-300] #  if index not in 
    for record_name in tqdm(train_record_names, desc="Reading train data: "):
        X_train.extend(read_pickle(os.path.join(train_dataset_folder, f"{record_name}_{sensor_name}_X_data.pkl")))
        y_train.extend(read_pickle(os.path.join(train_dataset_folder, f"{record_name}_{sensor_name}_y_data.pkl")))
    
    for record_name in tqdm(validation_record_names, desc="Reading validation_data"):
        X_validation.extend(read_pickle(os.path.join(train_dataset_folder, f"{record_name}_{sensor_name}_X_data.pkl")))
        y_validation.extend(read_pickle(os.path.join(train_dataset_folder, f"{record_name}_{sensor_name}_y_data.pkl")))
        
    y_train = label_interceptor(y_train)
    y_validation = label_interceptor(y_validation)
    
    # X_train, y_train = SMOTE(sampling_strategy="minority").fit_resample(X_train, y_train)
    
    class_w = class_weight_obj.compute_class_weight('balanced', classes=np.unique(y_validation), y=y_validation)
    print("Validation dataset data ratio:", class_w)    
    # print(len(y_validation))
    # for index in range(len(y_validation)):
    #     if y_validation[index] == 1:
    #         print(X_validation[index].transpose())
    #         # exit(1)
    #         plt.plot(X_validation[index].transpose()[0])
    #         plt.show()
    # exit(1)
    return np.array(X_train), np.array(X_validation), np.array(y_train), np.array(y_validation)


def lr_schedule(epoch, lr):
    # if epoch > 70 and \
    #         (epoch - 1) % 10 == 0:
    #     lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def train(epochs=30):
    global_start_time = time.time()
    
    model = build_transformer_model((300,1), CLASSES_NUM)
    
    print ('\nData Loaded. Compiling...\n')
    print('Loading data... ')
    X_train, X_validation, y_train, y_validation= get_train_and_validate_data("ABDO")  # ABDO | THOR | NEW

    # feature engneering
    # print(X_train)
    # print(X_train.shape)
    # exit(1)
    # X_train = X_train.transpose((0, 2, 1))
    # X_validation = X_validation.transpose((0, 2, 1))

    class_w = class_weight_obj.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    print(class_w)
    
    y_train = to_categorical(y_train, CLASSES_NUM)
    y_validation = to_categorical(y_validation, CLASSES_NUM)

    try:
        print("Training...")
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        
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
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, 
                            class_weight=class_weight,
                            validation_data=(X_validation, y_validation),
                            # validation_split=0.1,
                            callbacks=[earlystop_callback, select_best_checkpoint, lr_scheduler])
        pd.DataFrame(history.history).to_csv('training_log.csv', index=False)

        print(f"Save model to {model_filepath} success")

    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
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
    train(epochs=10)
    plot_history()
