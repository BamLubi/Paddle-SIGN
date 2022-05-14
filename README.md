# Detecting Beneficial Feature Interactions for Recommender Systems

![](https://img.shields.io/badge/-paddlepaddle-brightgreen)
![](https://img.shields.io/badge/-GAN-brightgreen)

> 本项目为百度飞桨论文复现挑战赛（第六期）论文复现项目。
>
> 完整的项目请移步：[PaddleRec-sign](https://github.com/PaddlePaddle/PaddleRec/blob/master/models/rank/sign/readme.md)

## 模型简介

特征交叉通过将两个或多个特征相乘，来实现样本空间的非线性变换，提高模型的非线性能力，其在推荐系统领域中可以显著提高准确率。以往的研究考虑了所有特征之间的交叉，但是某些特征交叉与推荐结果的相关性不大，其引入的噪声会降低模型的准确率。因此论文[《Detecting Beneficial Feature Interactions for Recommender Systems》]( https://arxiv.org/pdf/2008.00404v6.pdf )中提出了一种利用图神经网络自动发现有意义特征交叉的模型L0-SIGN。

作者使用图神经网络建模每个样本的特征，将特征交叉与图中的边相联系，用GNN的关系推理能力对特征交叉进行建模。使用L0正则化的边预测来限制图中检测的边的数量，以此进行有意义特征交叉的检测。

## 模型组网

L0-SIGN模型有两个模块，一个是L0边预估模块，通过矩阵分解图的邻接矩阵进行边的预估，一个是图分类SIGN模块。模型的主要组网结构如图1所示，与 net.py 中的代码一一对应 ：

<p align="center">
<img align="center" src="https://picgo-1256052225.cos.ap-guangzhou.myqcloud.com/img/202201241713641.png">
<p>

## 快速开始
1. 克隆PaddleRec项目至本地

   ```shell
   git clone https://github.com/PaddlePaddle/PaddleRec.git
   cd PaddleRec
   ```

2. 将本项目覆盖PaddleRec目录

3. 安装环境

   ```shell
   pip install -r requirements.txt
   ```

4. 参考[模型简介](./models/rank/sign/README.md)，以运行项目
