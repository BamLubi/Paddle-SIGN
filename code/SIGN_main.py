import enum
from json import load
from dataloader import Dataset
import argparse
from torch_geometric.data import DataLoader
from SIGN_train import train



## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-tag-test', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--l0_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size 1024')
parser.add_argument('--n_epoch', type=int, default=1, help='the number of epochs 500')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), \
                            zeta (interval_min) and gama (interval_max).")
parser.add_argument('--hidden_layer', type=int, default=32, help='neural hidden layer 32')
parser.add_argument('--pred_edges', type=int, default=1, help='!=0: use edges in dataset, 0: predict edges using L_0')
parser.add_argument('--random_seed', type=int, default=2022, help='size of common item be counted')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
args = parser.parse_args()

# 获取数据集
dataset = Dataset('../data', args.dataset, pred_edges=args.pred_edges)
num_feature = dataset.node_M() 
data_num = dataset.data_N()


# 划分数据集(0.7,0.15,0.15)
dataset.shuffle()
train_index = int(len(dataset)* 0.7)
test_index = int(len(dataset) * 0.85)
train_dataset = dataset[:train_index]
test_dataset = dataset[train_index:test_index]
val_dataset = dataset[test_index:]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

show_loss = True
print(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
""")

## 训练
datainfo = [train_loader, val_loader, test_loader, num_feature]
train(args, datainfo, show_loss, [len(train_dataset), len(val_dataset), len(test_dataset)])