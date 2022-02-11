from dataloader import RandomDataset, collate_fn
from paddle.io import DataLoader, random_split
import argparse
from SIGN_train import train

## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-tag', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--l0_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size 1024')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs 500')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), \
                            zeta (interval_min) and gama (interval_max).")
parser.add_argument('--hidden_layer', type=int, default=32, help='neural hidden layer 32')
parser.add_argument('--pred_edges', type=int, default=1, help='!=0: use edges in dataset, 0: predict edges using L_0')
parser.add_argument('--random_seed', type=int, default=2022, help='size of common item be counted')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
args = parser.parse_args()


dataset = RandomDataset(dataset="../data/"+args.dataset+".data", pred_edges=args.pred_edges)
num_feature = dataset.node_M() 
data_num = dataset.data_N()
num_graphs = dataset.get_num_graph()

# # test
# loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
# for i,data in enumerate(loader):
#     # data['node_attr'] = data['node_attr'].reshape([2*3,1])
#     # print(data['edge_attr'].shape, data['edge_attr'].dtype, type(data['edge_attr']))
#     # print(data['edge_attr'])
#     # tmp = data['edge_attr'].reshape([2*6,2])
#     # print(data['edge_attr'].shape, data['edge_attr'].dtype, type(data['edge_attr']))
#     # print(data['edge_attr'])
    
#     # data['edges'] = data['edges'].reshape([2*6,2])
#     # data['num_nodes'] = data['num_nodes'].reshape([2])
#     # data['num_edges'] = data['num_edges'].reshape([2])
#     # g = pgl.Graph(
#     #     edges=data['edges'],
#     #     num_nodes=data['num_nodes'],
#     #     num_edges=data['num_edges'],
#     #     node_feat={'node_attr': data['node_attr']},
#     #     edge_feat={'edge_attr': data['edge_attr']}
#     # )
#     # print(g)
    
#     # print("graph_node_id", g.graph_node_id)
#     # print("edges", g.edges)
#     # print("node_attr", g.node_feat['node_attr'])
#     # print("edge_attr", g.edge_feat['edge_attr'])
#     print("i:", i)
#     # print("edges", data['edges'])
#     # print("num_nodes", data['num_nodes'])
#     # print("num_edges", data['num_edges'])
#     # print("node_attr", data['node_attr'])
#     # print("edge_attr", data['edge_attr'])
#     print("label", data['label'])
#     break

# 划分数据集(0.8,0.1,0.1)
train_index = int(len(dataset)* 0.8)
test_index = int(len(dataset) * 0.9) - train_index
val_index = len(dataset) - train_index - test_index
dataset = random_split(dataset, [train_index, test_index, val_index])
train_dataset, test_dataset, val_dataset = dataset[0], dataset[1], dataset[2]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

print(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
num_feature: {num_feature},
data_num: {data_num},
train:test:val: {[train_index,test_index,val_index]}
""")

# 训练
datainfo = [train_loader, test_loader, val_loader, num_feature, num_graphs]
train(args, datainfo, [len(train_dataset), len(val_dataset), len(test_dataset)])