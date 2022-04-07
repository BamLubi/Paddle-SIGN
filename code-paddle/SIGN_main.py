from dataloader import SIGNDataset, collate_fn
from paddle.io import DataLoader, random_split
import argparse
from SIGN_train import train
import logging
import sys

# 设置日志属性
rf_handler = logging.StreamHandler(sys.stderr)
rf_handler.setLevel(logging.DEBUG) 
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))

f_handler = logging.FileHandler('./train.log', mode='a')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[rf_handler, f_handler])

## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-tag', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--l0_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size 1024')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs 500')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]', help="l0 parameters, which are beta (temprature), zeta (interval_min) and gama (interval_max).")
parser.add_argument('--hidden_layer', type=int, default=32, help='neural hidden layer 32')
parser.add_argument('--pred_edges', type=int, default=1, help='!=0: use edges in dataset, 0: predict edges using L_0')
parser.add_argument('--random_seed', type=int, default=2022, help='size of common item be counted')
parser.add_argument('--device', type=str, default='gpu', help='whether to use gpu')
args = parser.parse_args()


dataset = SIGNDataset(dataset="../data/"+args.dataset+".data", pred_edges=args.pred_edges)
num_node = dataset.node_num
num_edge = dataset.edge_num
num_graph = dataset.graph_num
num_feature = dataset.feature_num

# 划分数据集(0.8,0.1,0.1)
train_index = int(len(dataset)* 0.8)
test_index = int(len(dataset) * 0.9) - train_index
val_index = len(dataset) - train_index - test_index
dataset = random_split(dataset, [train_index, test_index, val_index])
train_dataset, test_dataset, val_dataset = dataset[0], dataset[1], dataset[2]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

logging.info(f""" Train Config:
datast: {args.dataset},
vector-dim: {args.dim},
batch_size: {args.batch_size},
lr: {args.lr},
n_epoch: {args.n_epoch},
hidden_layer: {args.hidden_layer},
l0_weight: {args.l0_weight},
l2_weight: {args.l2_weight},
feature: {num_feature},
graphs: {num_graph},
nodes: {num_node},
edge: {num_edge},
train:val:test: {[len(train_dataset), len(val_dataset), len(test_dataset)]}
""")

# 训练
datainfo = [train_loader, test_loader, val_loader, num_feature]
train(args, datainfo, [len(train_dataset), len(val_dataset), len(test_dataset)])