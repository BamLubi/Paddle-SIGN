# Detecting Beneficial Feature Interactions for Recommender Systems

 以下是本例的简要目录结构及说明： 

```shell
├── data # sample数据
    ├── ml-tag-128k.data
├── datasets # 全量数据
    ├── ml-tag.data
├── test_tipc # TIPC
├── __init__.py 
├── README.md # 文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── sign_reader.py # 数据读取程序
├── net.py # 模型核心组网（动静统一）
├── dygraph_model.py # 构建动态图
├── train.log # 全量训练和预测日志
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介

特征交叉通过将两个或多个特征相乘，来实现样本空间的非线性变换，提高模型的非线性能力，其在推荐系统领域中可以显著提高准确率。以往的研究考虑了所有特征之间的交叉，但是某些特征交叉与推荐结果的相关性不大，其引入的噪声会降低模型的准确率。因此论文[《Detecting Beneficial Feature Interactions for Recommender Systems》]( https://arxiv.org/pdf/2008.00404v6.pdf )中提出了一种利用图神经网络自动发现有意义特征交叉的模型L0-SIGN。

作者使用图神经网络建模每个样本的特征，将特征交叉与图中的边相联系，用GNN的关系推理能力对特征交叉进行建模。使用L0正则化的边预测来限制图中检测的边的数量，以此进行有意义特征交叉的检测。

## 数据准备

论文使用了4个开源数据集，`DBLP_v1`、`frappe`、`ml-tag`、`twitter`，这里使用`ml-tag`验证模型效果，在模型目录的data目录中准备了快速运行的示例数据，在datasets目录中准备了全量数据。
该数据集专注于电影标签推荐，每个数据实例都代表一个图，数据格式如下：

```shell
# 电影标签 用户ID 电影ID 电影ID
0.0 24 25 26
1.0 62 63 64
```

## 运行环境

PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在sign模型目录的快速执行命令如下： 
```bash
# 拷贝本项目至PaddleRec项目的models文件夹
# git clone https://github.com/PaddlePaddle/PaddleRec/
# mv Paddle-SIGN PaddleRec/models/sign
# 进入模型目录
# cd PaddleRec/models/sign # 在任意目录均可运行
# 动态图训练
python -u ../../tools/trainer.py -m config.yaml # sample数据运行
python -u ../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行
# 动态图预测
python -u ../../tools/infer.py -m config.yaml # sample数据预测
python -u ../../tools/infer.py -m config_bigdata.yaml # 全量数据预测
```

## 模型组网
L0-SIGN模型有两个模块，一个是L0边预估模块，通过矩阵分解图的邻接矩阵进行边的预估，一个是图分类SIGN模块。模型的主要组网结构如图1所示，与 net.py 中的代码一一对应 ：

<p align="center">
<img align="center" src="https://picgo-1256052225.cos.ap-guangzhou.myqcloud.com/img/202201241713641.png">
<p>


## 复现效果
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。全量训练和预测的日志文件请见 train.log。
在全量数据下模型的指标如下：

| 模型 | auc   | acc   | batch_size | epoch_num | Time of each epoch |
| :------| :------ | :------ | :------| :------ | :------ |
| SIGN | 0.9418 | 0.8927 | 1024 | 20 | 约18分钟 |

1. 确认您当前所在目录为PaddleRec/models/sign
2. 进入PaddleRec/sign/datasets目录下，执行该脚本，会从国内源的服务器上下载sign全量数据集，并解压到指定文件夹。
``` bash
cd ./datasets/sign
sh run.sh
```
3. 切回模型目录,执行命令运行全量数据
```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行
python -u ../../tools/infer.py -m config_bigdata.yaml # 全量数据预测
```

## 进阶使用

## FAQ
