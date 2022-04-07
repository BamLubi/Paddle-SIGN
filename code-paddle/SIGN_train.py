from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import time
import paddle
import pgl
from SIGN_model import L0_SIGN
import logging

def train(args, data_info, data_nums):
    paddle.device.set_device(args.device)
    train_loader = data_info[0]
    test_loader = data_info[1]
    val_loader = data_info[2]
    num_feature = data_info[3]
    
    model = L0_SIGN(args, num_feature)
    
    optimizer = paddle.optimizer.Adagrad(
        learning_rate=args.lr,
        parameters=model.parameters(),
        epsilon=1e-05,
        weight_decay=1e-05
    )
    
    crit = paddle.nn.MSELoss()
    
    loss_list = []
    ACC_list = []
    AUC_list = []

    logging.info('Start training...')
    start_time = time.time()
    for step in range(args.n_epoch):
        # 训练
        loss_all = 0
        model.train()
        for data in train_loader:
            g, label = data
            g = pgl.Graph.batch(g).tensor()
            label = paddle.to_tensor(label, dtype='float32')
            
            output, l0_penaty, l2_penaty, num_edges = model(g)
            
            baseloss = crit(output, label)
            l0_loss = l0_penaty * args.l0_weight 
            l2_loss = l2_penaty * args.l2_weight
            loss = baseloss + l0_loss + l2_loss 
            loss_all += g.num_graph * loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        cur_loss = loss_all / data_nums[0]
        # 验证
        train_auc, train_acc, train_edges = evaluate(model, train_loader)
        val_auc, val_acc, val_edges = evaluate(model, val_loader)
        test_auc, test_acc, test_edges = evaluate(model, test_loader)
        # 打印信息
        end_time = time.time()
        t = end_time - start_time
        logging.info('Epoch: {:03d}, Loss: {:.3f}, Train Auc: {:.3f}, Acc: {:.3f}; Val Auc: {:.3f}, Acc: {:.3f}; Test Auc: {:.3f}, Acc: {:.3f}; Train time: {:.2f}'.
            format(step, cur_loss.item(), train_auc, train_acc, val_auc, val_acc, test_auc, test_acc, t))
        start_time = time.time()
        # 记录数据
        loss_list.append(cur_loss.item())
        ACC_list.append(val_acc)
        AUC_list.append(val_auc)
    
    # 绘制图形数据
    draw_plot(loss_list, ACC_list, AUC_list)

def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = []
    edges_all = 0
    with paddle.no_grad():
        for data in loader:
            g, label = data
            g = pgl.Graph.batch(g).tensor()
            label = paddle.to_tensor(label, dtype='float32')
            
            pred, _, _, num_edges = model(g)
            pred = pred.detach().cpu().numpy()
            edges_all += num_edges
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    labels = labels[:,1].reshape((-1,1))
    
    m = paddle.metric.Auc()
    m.update(preds=predictions, labels=labels)
    auc = m.accumulate()
    
    m = paddle.metric.Accuracy()
    correct = m.compute(paddle.to_tensor(predictions), paddle.to_tensor(labels))
    m.update(correct)
    acc = m.accumulate()
    
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
    
    save_path = 'epoch_{:d}_acc_{:.3f}.jpg' . format(len(ACC), ACC[-1])
    plt.savefig(save_path)
    logging.info("Save plot to :" + save_path)
    
    plt.show()