from operator import imod
import numpy as np
import paddle
from paddle.io import Dataset

class RandomDataset(Dataset):
    def __init__(self, dataset="../data/ml-tag.data", pred_edges=1):
        self.dataset = dataset
        self.pred_edges = pred_edges
        
        self.data_list = []
        
        self.node_num = 0
        self.data_num = 0
        self.num_graphs = 0
        
        self.process()
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)

    def shuffle(self):
        np.random.shuffle(self.data_list)

    def process(self):
        node, edge, label, sr_list, self.node_num, self.data_num = self.read_data()
        self.num_graph = len(node)
        
        for i in range(len(node)):
            num_nodes = len(node[i])
            num_nodes = paddle.to_tensor(num_nodes, dtype='int64')
            node_features = paddle.to_tensor(node[i], dtype='int64')
            node_features = paddle.unsqueeze(node_features, axis=1) # 维度＋1 => 二维
            
            edges = []
            for u,v in zip(edge[i][0],edge[i][1]):
                u_v = (u,v)
                edges.append(u_v)
            edges = paddle.to_tensor(edges, dtype='int64')
            num_edges = len(edges)
            num_edges = paddle.to_tensor(num_edges, dtype='int64')
            # edge_index = paddle.to_tensor(edge[i], dtype='int64')
            
            y = paddle.to_tensor(label[i], dtype='float32')
            
            if self.pred_edges:
                sr = paddle.to_tensor(sr_list[i], dtype='int64')
            else:
                sr = []
            
            tmp = {
                "edges": edges,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "node_attr": node_features,
                "edge_attr": sr,
                "label": y
            }
            # g = pgl.Graph(
            #     edges=edges,
            #     num_nodes=num_nodes,
            #     num_edges=num_edges,
            #     node_feat={'node_attr': node_features},
            #     edge_feat={'edge_attr': sr}
            # )
            self.data_list.append(tmp)
        # 后续可以加上tensor的保存
        # torch.save((data, slices), self.processed_paths[0])
   
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
            sr_list = []    #sender_receiver_list, containing node index
            for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                sr_list.append(sr_l)

        # 将label转换成onehot编码
        label = self.construct_one_hot_label(label)

        return node_list, edge_list, label, sr_list, max_node_index + 1, data_num
    
    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]]                 # [[sender...], [receiver...]]
        sender_receiver_list = []           # [[s,r],[s,r]...]
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
    
    def node_M(self):
        """获取最大的节点id"""
        return self.node_num
    
    def data_N(self):
        """获取数据数量"""
        return self.data_num
    
    def get_num_graph(self):
        """获取数据数量"""
        return self.num_graph
    
def collate_fn(batch_data):
    edges = None
    num_nodes = None
    num_edges = None
    node_attr = None
    edge_attr = None
    label = None

    for i,data in enumerate(batch_data):
        edges = paddle.concat([edges, data['edges']]) if isinstance(edges, paddle.Tensor) else data['edges'].clone()
        num_nodes = num_nodes + data['num_nodes'] if isinstance(num_nodes, paddle.Tensor) else data['num_nodes'].clone()
        num_edges = num_edges + data['num_edges'] if isinstance(num_edges, paddle.Tensor) else data['num_edges'].clone()
        node_attr = paddle.concat([node_attr, data['node_attr']]) if isinstance(node_attr, paddle.Tensor) else data['node_attr'].clone()
        edge_attr = paddle.concat([edge_attr, data['edge_attr']]) if isinstance(edge_attr, paddle.Tensor) else data['edge_attr'].clone()
        if data['label'].shape != [1,2]:
            data['label'] = data['label'].unsqueeze(0)
        label = paddle.concat([label, data['label']]) if isinstance(label, paddle.Tensor) else data['label'].clone()

    batch_data = {
        "edges": edges,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "label": label
    }
    return batch_data