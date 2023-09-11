# 基于AI的睡眠呼吸暂停检测🧑‍⚕️

>药物能治疗疾病，但只有医生能治疗患者。- Carl Jung



## 目标





## 目录

[TOC]

## 文件及文件夹说明

### ✨核心文件夹

- #### data/

  存储来自不同数据集中的数据，常用的公开数据集有SHHS和MESA，此处以MESA数据集为例进行说明

  * mesa/

    + label_data/

      从XML格式文件中提取的存储标签事件，每个事件包括：事件开始事件、事件持续时间、事件类型、判断依据的设备

    + raw_data/

      从EDF格式文件中提取的原始传感器数据，每个文件对应一个字典，包括：数据采样率、一维时序数据

    + source_edfs/

      以EDF格式保存的原始传感器数据，直接从原始MESA数据集中拷贝而来

    + source_labels/

      以XML格式保存的原始标签事件数据，直接从原始MESA数据集中拷贝而来

    + test_data/

      经过处理后用于测试的数据

    + train_data/

      经过处理后用于训练的数据

    + validation_data/

      经过处理后用于训练过程中验证的数据



- #### deployment/

  存储与模型布署相关的脚本

  * onnx_tool.py
    1. 将使用TensorFlow训练的h5格式的模型文件转换为ONNX格式的模型文件
    2. 加载ONNX格式的模型文件进行预测



- #### models/

  存储模型的具体实现，以TensorFlow框架下的Transformer模型为例

  - transformer.py

    实现构造模型的build_xxx_model方法，返回tf.keras.Model类型的对象，供训练和测试脚本调用



- #### notebooks/

  存储Jupyter Notebook、Jupyter Lab等格式的文件，用于进行数据分析实验等



- #### outputs/

  脚本运行过程中生成的一些文件，包括：数据集样本信息统计信息等

  - *_samples_SA_intensity_info.pkl

    每个样本的呼吸暂停严重程度信息



- #### trained_models/

  存储训练好的模型文件





### ✨核心文件

- #### config.py

  使用`dynaconf init -f toml`生成，用于其他脚本获取全局配置信息

- #### settings.toml

  使用`dynaconf init -f toml`生成，用于存储全局配置信息

  ```toml
  [shhs1]
  # SHHS-1数据集中的原始传感器数据文件夹路径
  source_edf_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\source_edfs"
  
  # 从原始EDF格式传感器数据中提取的字典格式传感器数据文件夹路径，包括：采样率、一维时序数据
  raw_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\raw_data"
  
  # SHHS-1数据集中的原始呼吸暂停事件标签数据文件夹数据
  source_sleep_apnea_label_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\source_labels"
  
  # 从原始XML格式呼吸暂停事件标签数据中提取的字典格式呼吸暂停事件数据文件夹路径，包括每个事件的：开始事件、结束时间、事件类型、判断依据的设备
  sleep_apnea_label_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\label_data"
  
  # SHHS-1数据集中的样本患呼吸暂停综合征的统计信息
  samples_SA_intensity_info_filename = "shhs1_all_samples_SA_intensity_info.pkl"
  
  # 经过处理后用于训练模型的数据所在的文件夹
  train_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\train_data"
  
  # 经过处理后用于训练过程中验证的数据所在的文件夹
  validation_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\shhs1\\validation_data"
  
  # 经过处理后用于测试的数据所在的文件夹
  test_data_path = "YourProjectDirectory\\PythonProject\\sleep_apnea_detection\\data\\shhs1\\test_data"
  
  [mesa]
  source_edf_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\source_edfs"
  raw_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\raw_data"
  source_sleep_apnea_label_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\source_labels"
  sleep_apnea_label_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\label_data"
  samples_SA_intensity_info_filename = "mesa_all_samples_SA_intensity_info.pkl"
  train_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\train_data"
  validation_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\validation_data"
  test_data_path = "YourProjectDirectory\\sleep_apnea_detection\\data\\mesa\\test_data"
  
  [common_filepath]
  output_root_folder = "YourProjectDirectory\\sleep_apnea_detection\\outputs"
  
  [preprocess]
  resample_rate = 10
  window_length_secs = 10
  window_slide_stride_secs = 4
  sleep_apnea_overlap_length_secs = 4
  
  low_pass_cutoff = 0.7
  low_pass_filter_order = 2
  high_pass_cutoff = 0.2
  high_pass_filter_order = 2
  
  [train]
  trained_models_root_folder = "YourProjectDirectory\\sleep_apnea_detection\\trained_models"
  model_name = "cnn_transformer"
  class_num = 2
  epochs = 10
  batch_size = 128
  learning_rate = 0.001
  ```

  

- #### display_data.py

- #### evaluate.py

- #### extract_signals_from_edf.py

- #### extract_sleep_apnea_events_from_xml.py

- #### prepare_train_test_data.py

- #### select_samples.py

- #### sleep_data_obj.py

- #### train.py

- #### utils.py

- #### README.md

- #### README-cn.md

- #### requirements.txt





### 隐藏文件夹

- #### .git/

  git相关信息



- #### .vscode/

  vscode相关配置信息





### 隐藏文件

- #### .gitignore

  无需同步到git中的文件/文件夹





### 其他文件夹

- #### \__pacache__/

  Python解释器生成的文件







## 主要功能及使用方法

### 原始数据集处理





### 数据样本可视化





### 训练、验证、测试数据集的构建



