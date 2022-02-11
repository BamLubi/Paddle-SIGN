import numpy as np
from SIGN_model import L0_SIGN
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import paddle
import pgl
import matplotlib.pyplot as plt


def train(args, data_info, data_nums):
    train_loader = data_info[0]
    test_loader = data_info[1]
    val_loader = data_info[2]
    num_feature = data_info[3]
    num_graphs = data_info[4]
    
    model = L0_SIGN(args, num_feature)
    
    optimizer = paddle.optimizer.Adagrad(learning_rate=args.lr,
                                         parameters=model.parameters(),
                                         epsilon=1e-05,
                                         weight_decay=1e-05)
    
    crit = paddle.nn.MSELoss()
    
    loss_list = []
    ACC_list = []
    AUC_list = []

    print('start training...')
    start_time = time.time()
    for step in range(args.n_epoch):
        # training
        loss_all = 0
        edge_all = 0
        model.train()
        for data in train_loader:
            label = data['label']
            g = graph2tensor(data, args.batch_size)
            
            output, l0_penaty, l2_penaty, num_edges = model(g)
            
            baseloss = crit(output, label)
            l0_loss = l0_penaty * args.l0_weight 
            l2_loss = l2_penaty * args.l2_weight
            loss = baseloss + l0_loss + l2_loss 
            loss_all += num_graphs * loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        cur_loss = loss_all / data_nums[0]

        # evaluation
        # train_auc, train_acc, _ = evaluate(model, train_loader, args.batch_size)
        train_auc = 0
        train_acc = 0
        val_auc, val_acc, _ = evaluate(model, val_loader, args.batch_size)    
        test_auc, test_acc, test_edges = evaluate(model, test_loader, args.batch_size)

        end_time = time.time()
        t = end_time - start_time
        print('Epoch: {:03d}, Loss: {:.4f}, Train Auc: {:.4f}, Train Acc: {:.4f}, Val Auc: {:.4f}, Acc: {:.4f}, Test Auc: {:.4f}, Acc: {:.4f}, Train edges: {:07d}, Train time: {:.2f}'.
          format(step, cur_loss, train_auc, train_acc, val_auc, val_acc, test_auc, test_acc, test_edges, t))
        start_time = time.time()
        ## prepare plot data
        loss_list.append(cur_loss)
        ACC_list.append(val_acc)
        AUC_list.append(val_auc)
    # draw plot
    draw_plot(loss_list, ACC_list, AUC_list)

def graph2tensor(data, batch_size):
    # data['node_attr'] = data['node_attr'].reshape([batch_size*3,1])
    # data['edge_attr'] = data['edge_attr'].reshape([batch_size*6,2])
    # data['edges'] = data['edges'].reshape([batch_size*6,2])
    # data['num_nodes'] = data['num_nodes'].reshape([batch_size])
    # data['num_edges'] = data['num_edges'].reshape([batch_size])
    g = pgl.Graph(
        edges=data['edges'],
        num_nodes=data['num_nodes'],
        node_feat={'node_attr': data['node_attr']},
        edge_feat={'edge_attr': data['edge_attr']}
    )
    # 加下面这个转换好像出问题，是不是元素已经是tensor的缘故
    # g = pgl.Graph.batch(g).tensor()
    return g

def evaluate(model, loader, batch_size):
    model.eval()
    predictions = []
    labels = []
    edges_all = 0
    with paddle.no_grad():
        for data in loader:
            label = data['label']
            g = graph2tensor(data, batch_size)
            
            pred, _, _, num_edges = model(g)
            pred = pred.detach().cpu().numpy()
            edges_all += num_edges
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(np.argmax(labels, 1), np.argmax(predictions, 1))
    return auc, acc, edges_all

def draw_plot(loss, ACC, AUC):
    
    plt.figure(figsize=(12,6), dpi=80)
    plt.figure(1)
    plt.rcParams['font.sans-serif']='SimHei'

    ax1 = plt.subplot(121)
    plt.plot(range(len(loss)), loss, color='b', label='LOSS')
    # plt.plot(range(len(loss)), loss, color='b', marker='o', label='LOSS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("LOSS")
    plt.legend(loc='best')

    ax2 = plt.subplot(122)
    plt.plot(range(len(ACC)), ACC, color='r', label='ACC')
    plt.plot(range(len(AUC)), AUC, color='g', label='AUC')
    # plt.plot(range(len(ACC)), ACC, color='r', marker='s', label='ACC')
    # plt.plot(range(len(AUC)), AUC, color='g', marker='d', label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ACC | AUC')
    plt.title("ACC & AUC")
    plt.legend(loc='best')

    plt.show()