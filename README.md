## Environmental Requirement

```shell
# 1. set up envirnment
$ conda create -n SIGN python=3.7.2
$ conda activate SIGN

# 2. install torch, torchvision, 
$ pip install torch==1.6.0 torchvision==0.7.0
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-geometric
# 3. install requirements
$ pip install -r requirements.txt
```

## Run the code

   ```shell
   $ cd code
   $ python SIGN_main.py
   ## default params:
   ## dataset: ml-tag
   ## batch_size: 1024
   ## n_epoch: 500
   ## hidden_layer: 32
   ## lr: 0.05
   ```

[模型分析]:./SIGN算法分析笔记.md
[论文]:./SIGN.pdf

