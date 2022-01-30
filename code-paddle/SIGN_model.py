import numpy as np
import paddle
import paddle.nn.functional as F
import pgl


class L0_SIGN(paddle.nn.Layer):
    def __init__(self, args, n_feature):
        super(L0_SIGN, self).__init__()

        self.pred_edges = args.pred_edges
        self.n_feature = n_feature
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.l0_para = eval(args.l0_para)
        self.batch_size = args.batch_size

        if self.pred_edges:
            self.linkpred = LinkPred(self.dim, self.hidden_layer, self.n_feature,  self.l0_para)
            # self.linkpred = self.linkpred.to(self.device)

        self.sign = SIGN(self.dim, self.hidden_layer)
        self.g = paddle.nn.Linear(self.dim, 2) # 2 is the class dimention  
        
        self.feature_emb = paddle.nn.Embedding(self.n_feature, self.dim)
        
        self.add_sublayer("g", self.g)


    def forward(self, graph, is_training=True):
        # does not conduct link prediction, use all interactions
        # graph: pgl.Graph object
        # graph.node_feat['node_attr']: [bacth_size*3, 1]
        # graph.edge_feat['edge_attr']: [bact_size*6, 2]
        # graph.edges: [bact_size*6, 2]
        
        # x, edge_index, sr = data['x'], data['edge_index'], data['edge_attr']
        
        x, edge_index, sr = graph.node_feat['node_attr'], graph.edges, graph.edge_feat['edge_attr']
        # segment_ids = graph.graph_node_id
        segment_ids = [int(i/3) for i in range(self.batch_size * 3)]
        segment_ids = paddle.to_tensor(segment_ids, dtype='int64')

        x = self.feature_emb(x)
        # x = paddle.squeeze(x, axis=1)
        x = x.squeeze(1)

        if self.pred_edges:
            sr = paddle.transpose(sr, perm=[1,0])
            s, l0_penaty = self.linkpred(sr, is_training)
            pred_edge_index, pred_edge_weight = self.construct_pred_edge(edge_index, s) 
            
            graph = pgl.Graph(
                node_feat={'node_attr': x},
                edges=pred_edge_index
            )
            # print("调用sign前","----"*10)
            # print("edge_index", edge_index.shape)
            # print("pred_edge_weight", pred_edge_weight.shape)
            # print("x==node_features", x.shape)
            # print("sr", sr.shape)
            # print("----"*10)
            updated_nodes = self.sign(graph, x, pred_edge_weight)
            num_edges = pred_edge_weight.shape[0]
        else:
            updated_nodes = self.sign(graph, x, sr)
            l0_penaty = 0
            num_edges = edge_index.shape[1]
        
        l2_penaty = paddle.multiply(updated_nodes, updated_nodes).sum()
        
        
        updated_nodes = pgl.math.segment_mean(updated_nodes, segment_ids)
        out = self.g(updated_nodes)
        # out = out.reshape([self.batch_size*2, 1])
        # out = out.squeeze(1)
        
        # print("pgl.math.segment_mean","----"*10)
        # print("updated_nodes", updated_nodes.shape)
        # print("segment_ids", segment_ids.shape)
        # print("out", out.shape)
        # print("----"*10)
        
        return out, l0_penaty, l2_penaty, num_edges 

    def construct_pred_edge(self, fe_index, s):
        """
        fe_index: full_edge_index, [2, all_edges_batchwise]
        s: predicted edge value, [all_edges_batchwise, 1]

        construct the predicted edge set and corresponding edge weights
        """
        s = paddle.squeeze(s)
        fe_index = paddle.transpose(fe_index, perm=[1, 0])

        # s = s.numpy()
        # fe_index_np = fe_index.numpy()
        
        # sender = paddle.to_tensor(fe_index_np[0][s>0])
        # receiver = paddle.to_tensor(fe_index_np[1][s>0])
        # pred_weight = paddle.to_tensor(s[s>0])
        
        # sender = paddle.unsqueeze(sender, 0)
        # receiver = paddle.unsqueeze(receiver, 0)
        # pred_index = paddle.concat([sender, receiver], 0)
        
        # fe_index = paddle.to_tensor(fe_index)
        # fe_index = paddle.transpose(fe_index, perm=[1, 0])
        
        
        sender = paddle.unsqueeze(fe_index[0], 0)
        receiver = paddle.unsqueeze(fe_index[1], 0)
        pred_index = paddle.concat([sender, receiver], 0)
        pred_weight = s
        
        fe_index = paddle.transpose(fe_index, perm=[1, 0])

        return pred_index, pred_weight 


class SIGN(paddle.nn.Layer):
    def __init__(self, dim, hidden_layer, aggr_func="mean"):
        super(SIGN, self).__init__()
        self.aggr_func = "reduce_%s" % aggr_func

        #construct pairwise modeling network
        self.lin1 = paddle.nn.Linear(dim, hidden_layer)
        self.lin2 = paddle.nn.Linear(hidden_layer, dim)
        self.activation = paddle.nn.ReLU()
        
        self.add_sublayer("lin1_g", self.lin1)
        self.add_sublayer("lin2_g", self.lin2)
        self.add_sublayer("activation", self.activation)
    
    def _send_func(self, src_feat, dst_feat, edge_feat=None):
        pairwise_analysis = self.lin1(paddle.multiply(src_feat["src"],dst_feat["dst"]))
        pairwise_analysis = self.activation(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)

        if edge_feat != None:
            edge_feat_ = paddle.reshape(edge_feat["e_attr"],[-1,1])
            # interaction_analysis = pairwise_analysis * edge_feat
            interaction_analysis = paddle.multiply(pairwise_analysis, edge_feat_)
        else:
            interaction_analysis = pairwise_analysis
        # print("----"*10)
        # print("src_feat['src']", src_feat["src"].shape)
        # print("pairwise_analysis",pairwise_analysis.shape)
        # print("interaction_analysis", interaction_analysis.shape)
        # print("----"*10)
        return {'msg':interaction_analysis}

    def _recv_func(self, msg):
        return msg['msg']
        # return getattr(msg, self.aggr_func)(msg["msg"])

    def forward(self, graph, node_feature, edge_attr):
        """
        Args:
            graph: `pgl.Graph` instance.
            feature: A tensor with shape (num_nodes, input_size)
        Return:
            If `concat=True` then return a tensor with shape (num_nodes, hidden_size),
            else return a tensor with shape (num_nodes, hidden_size * num_heads) 
        """
        # print("补充前","----"*10)
        # print("node_feature", node_feature.shape, node_feature[:4, :])
        # print("edge_attr", edge_attr.shape)
        # print("----"*10)
        mis = edge_attr.shape[0] - node_feature.shape[0]
        if mis > 0:
            edge_attr = edge_attr[:node_feature.shape[0]]
            # for i in range(mis):
            #     node_feature = paddle.concat([node_feature, paddle.unsqueeze(node_feature[0],0)], 0)
        elif mis < 0:
            for i in range(-mis):
                edge_attr = paddle.concat([edge_attr, edge_attr[0]], 0)
            
        # print("----"*10)
        # print("node_feature", node_feature.shape)
        # print("edge_attr", edge_attr.shape)
        # print("----"*10)
        msg = self._send_func(src_feat={"src": node_feature.clone()},
            dst_feat={"dst": node_feature.clone()},
            edge_feat={"e_attr": edge_attr}
        )
        output = self._recv_func(msg)
        # output = graph.recv(reduce_func=self._recv_func, msg=msg)
        
        # msg = graph.send(
        #     self._send_func,
        #     src_feat={"src": node_feature.clone()},
        #     dst_feat={"dst": node_feature.clone()},
        #     edge_feat={"e_attr": edge_attr}
        # )
        # output = graph.recv(reduce_func=self._recv_func, msg=msg)


        return output


class LinkPred(paddle.nn.Layer):
    def __init__(self, D_in, H, n_feature, l0_para):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinkPred, self).__init__()
        self.linear1 = paddle.nn.Linear(D_in, H)
        self.linear2 = paddle.nn.Linear(H, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        
        self.add_sublayer("linear1_L",self.linear1)
        self.add_sublayer("linear2_L",self.linear2)
        self.add_sublayer("relu_L",self.relu)
        self.add_sublayer("dropout_L",self.dropout)
        
        with paddle.no_grad():
            self.linear2.weight.set_value(self.linear2.weight + 0.2)

        self.temp = l0_para[0]      #temprature
        self.inter_min = l0_para[1] 
        self.inter_max = l0_para[2] 
        
        self.feature_emb_edge = paddle.nn.Embedding(n_feature, D_in,
                                                    weight_attr=paddle.ParamAttr(name='emb_weight',
                                                                                 initializer=paddle.nn.initializer.Normal(mean=0.2, std=0.01)))    #D_in is the dimension size
        # 权重初始化
        # self.linear2.weight.set_value(paddle.normal(shape=self.feature_emb_edge.shape(), mean=0.2, std=0.01))
        # self.feature_emb_edge.weight.data.normal_(0.2,0.01)

    def forward(self, sender_receiver, is_training):
        #construct permutation input
        sender_emb = self.feature_emb_edge(sender_receiver[0,:])
        receiver_emb = self.feature_emb_edge(sender_receiver[1,:])
        # _input =sender_emb * receiver_emb # element wise product sender and receiver embeddings
        _input = paddle.multiply(sender_emb, receiver_emb)
        #loc = _input.sum(1)
        h_relu = self.dropout(self.relu(self.linear1(_input)))
        loc = self.linear2(h_relu)
        if is_training:
            u = paddle.rand(loc.shape, dtype=loc.dtype)
            u.stop_gradient = False
            logu = paddle.log2(u)
            logmu = paddle.log2(1-u)
            sum_log = loc + logu - logmu
            s = F.sigmoid(sum_log/self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
        else:
            s = F.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min

        s = paddle.clip(s, min=0, max=1)

        l0_penaty = F.sigmoid(loc - self.temp * np.log2(-self.inter_min/self.inter_max)).mean()

        # print("----"*10)
        # print("sender_receiver", sender_receiver)
        # print("sender_emb", sender_emb)
        # print("receiver_emb", receiver_emb)
        # print("_input", _input)
        # print("h_relu", h_relu)
        # print("loc", loc)
        # print("u", u)
        # print("sum_log", sum_log)
        # print("s", s)
        # print("----"*10)
        
        return s, l0_penaty