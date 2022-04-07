import numpy as np
import paddle
from paddle.io import Dataset
from tqdm import tqdm
import pgl
import logging
import os

class SIGNDataset(Dataset):
    def __init__(self, dataset="../data/ml-tag.data", pred_edges=1):
        self.dataset = dataset
        self.pred_edges = pred_edges
        
        self.data_list = []
        
        self.node_num = 0
        self.edge_num = 0
        self.graph_num = 0
        self.feature_num = 0
        
        self.graph_list = []
        self.label_list = []
        
        self.process()
    
    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index]
    
    def __len__(self):
        return len(self.graph_list)

    def process(self):
        node, edge, label, sr_list, self.feature_num, self.data_num = self.read_data()
        self.graph_num = len(node)
        
        logging.info("Processing data...")
        for i in tqdm(range(len(node))):
            num_nodes = len(node[i])

            node_features = np.array(node[i],dtype='int32').reshape(len(node[i]), 1)
            
            edges = []
            for u,v in zip(edge[i][0], edge[i][1]):
                u_v = (u,v)
                edges.append(u_v)
            
            num_edges = len(edges)
            
            self.label_list.append(label[i])
            
            sr = sr_list[i] if self.pred_edges else []

            g = pgl.Graph(
                num_nodes=num_nodes,
                edges=edges,
                node_feat={"node_attr": node_features},
                edge_feat={"edge_attr": sr}
            )
            
            self.graph_list.append(g)
            
            self.node_num += num_nodes
            self.edge_num += num_edges
   
    def read_data(self):
        """读取数据集
        返回:
        """
        node_list = []
        label = []
        max_node_index = 0 # 节点序号最大值
        data_num = 0 # 数据集个数
        with open(self.dataset, 'r') as f:
            for line in f:
                data_num += 1
                data = line.split()
                # 第一个元素是label
                label.append(float(data[0]))
                # 其余元素是节点
                int_list = [int(data[i]) for i in range(len(data))[1:]]
                node_list.append(int_list)
                if max_node_index < max(int_list):
                    max_node_index = max(int_list)

        if not self.pred_edges:
            edge_list = [[[],[]] for _ in range(data_num)]
            sr_list = []
            # handle edges
            with open(self.edgefile, 'r') as f:
                for line in f:
                    edge_info = line.split()
                    node_index = int(edge_info[0])
                    edge_list[node_index][0].append(int(edge_info[1]))
                    edge_list[node_index][1].append(int(edge_info[2]))
        else:
            edge_list = []
            sr_list = []
            for index, nodes in enumerate(node_list):
            # for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                sr_list.append(sr_l)
        # 将label转换成onehot编码
        label = self.construct_one_hot_label(label)
        return node_list, edge_list, label, sr_list, max_node_index + 1, data_num
    
    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]] # [[sender...], [receiver...]]
        sender_receiver_list = []  # [[s,r],[s,r]...]
        for i in range(num_node):
            for j in range(num_node)[i:]:
                edge_list[0].append(i)
                edge_list[1].append(j)
                sender_receiver_list.append([nodes[i],nodes[j]])
        return edge_list, sender_receiver_list
    
    def construct_one_hot_label(self, label):
        """
        将label值转换成one-hot编码,label的取值属于{0,1}
        输入:[0,1,0,1]
        输出:[[1,0] [0,1] [1,0] [0,1]]
        """
        nb_classes = int(max(label)) + 1
        targets = np.array(label, dtype=np.int32).reshape(-1)
        return np.eye(nb_classes)[targets]
    
def collate_fn(batch_data):
    graphs = []
    labels = []
    for g, l in batch_data:
        graphs.append(g)
        labels.append(l)

    labels = np.array(labels, dtype="float32")

    return graphs, labels