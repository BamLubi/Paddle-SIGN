
# 飞桨训推一体认证（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleRec中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="./doc/guide.png" width="1000">
</div>


## 2. 测试工具简介
### 目录介绍

```shell
test_tipc/
├── configs/	# 配置文件目录
    ├── sign	# sign模型的测试配置文件目录
        ├── train_infer_python.txt	# 测试Linux上python训练预测（基础训练预测）的配置文件
├── results/	# 结果
├── output/		# 测试结果日志
├── test_train_inference_python.sh	# 测试python训练预测的主程序
└── readme.md	# 使用文档
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：


1. 运行测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

例如，测试基本训练预测功能的`lite_train_lite_infer`模式，运行：
```shell
# 运行测试
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/sign/train_infer_python.txt 'lite_train_lite_infer'
```
