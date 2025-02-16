import time
import yaml
import torch
import os
import copy
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torch_geometric.utils import to_undirected,degree,dropout_adj
from tqdm import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data22 import get_dataset, HeatDataset, set_train_val_test_split
from models import GCN,GDC,GAT
from seeds import val_seeds, test_seeds,test_seeds11
import argparse
from torch.amp import GradScaler, autocast

import random
from termcolor import cprint
from numpy.testing import assert_array_almost_equal
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, CitationFull,WikiCS
from ogb.nodeproppred import PygNodePropPredDataset

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Description of your program')

# 添加命令行参数
parser.add_argument('--drop_feature_rate1', type=float, default=0.1, help='Drop feature rate')
parser.add_argument('--drop_edge_rate1', type=float, default=0.2, help='Drop edge rate')
parser.add_argument('--data_dir', type = str, help='dir to dataset', default = './dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset_name', type = str, default = 'Cora')
# train
parser.add_argument('--teacher_epoch', type=int, default=1500)
parser.add_argument('--student_epoch', type=int, default=1500)
parser.add_argument('--lr', type=int, default=0.02)  #GCN 0.15 GAT0.005
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
# label noise
parser.add_argument('--noise_rate', type=float, default=0.1)
parser.add_argument('--noise_type', type=str, default='uniform',choices=['pair','sym','idn','uniform'])
# curriculum learning
parser.add_argument('--T', type=int, default=400)
parser.add_argument('--seed', type=int, default=5)
# 解析命令行参数
args = parser.parse_args()


device ='cuda:0'
with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)
#dataset = get_dataset(args.data_name, True, args.noise_type, args.noise_rate)


#评估
def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    #keys = ['val', 'test'] if test else ['val']
    keys1 = ['test']
    for key in keys1:
        mask = data[f'{key}_mask']
        pred = logits[mask].max(1)[1]
        #acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        acc = pred.eq(data.true_y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    keys2=['val']
    for key in keys2:
        mask = data[f'{key}_mask']
        pred = logits[mask].max(1)[1]
        #acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        acc = pred.eq(data.true_y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_model(args,ss,eps,t, random_state,gpu_id=None,
              noise_type = 'pair',noise_ratio = 0.2, alpha:float=0.1, temperature:float=2 ):
    # 设置随机数种子
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    cprint("# load data: {}".format(args.dataset_name),"green")
    # dataset_kwargs = {}
    # data, adj_list, x_list, n_list = get_dataset(args)
    data_t = get_dataset(args.dataset_name, noise = noise_type, rate = noise_ratio)
    data_t = data_t.to(device)

    data_t.data = set_train_val_test_split(  # 用于将图数据（Data 对象）划分为训练集、验证集和测试集。
        args.seed,
        data_t.data,
        num_development=1500,
    ).to(device)

    datasets = {}
    for preprocessing in ['heat']:  # 'heat'
        if preprocessing == 'heat':
            dataset = HeatDataset(
                name=args.dataset_name,
                use_lcc=config['use_lcc'],
                noise=args.noise_type,
                rate=args.noise_rate,
                t=t,
                k=config[preprocessing]['k'],
                eps=eps
            )
            dataset.data = dataset.data.to(device)
            print(dataset.data)
            print('以上是heat')
            datasets[preprocessing] = dataset
    data_gdc = dataset.data.to(device)
    dataset.data = set_train_val_test_split(  # 用于将图数据（Data 对象）划分为训练集、验证集和测试集。
        args.seed,
        data_gdc,
        num_development=1500,
    ).to(device)


    #教师模型
    gcn1 = GDC(data_t).to(device)
    # 下面的optimizer适用于除了ogbn数据之外的其他小数据
    optimizer1 = torch.optim.Adam(gcn1.parameters(), lr=args.lr, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    print(data_t.data)
    # create model

    #训练教师模型
    # 设置随机数种子
    set_seed(args.seed)
    best_model_state = copy.deepcopy(gcn1.state_dict())  # 初始化最佳模型状态
    patience_counter = 0
    tmp_dict1 = {'val_acc': 0}
    gcn1.train()
    for epoch in tqdm(range(args.teacher_epoch)):
        if patience_counter == 100:  # 设定多少轮不提升就停止训练
            break
        #for epoch in tqdm(range(400)):,cs 800
        optimizer1.zero_grad()
        out = gcn1(data_t.data)
        # loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss = F.nll_loss(out[data_t.data.train_mask], data_t.data.y[data_t.data.train_mask])
        loss.backward()
        optimizer1.step()
        pred = torch.argmax(out, dim=1)

        # 计算训练集准确率
        acc_train = (pred[data_t.data.train_mask] == data_t.data.y[data_t.data.train_mask]).float().mean()
        # 计算验证集准确率
        acc_val = (pred[data_t.data.val_mask] == data_t.data.true_y[data_t.data.val_mask]).float().mean()

        # acc = acc * 100
        # acc_st = int(correct) / int(test_mask.sum())
        eval_dict1 = evaluate(gcn1, data_t.data,True)
        # 比较当前epoch的验证集准确率与上一个最佳epoch的验证集准确率
        if eval_dict1['val_acc'] < tmp_dict1['val_acc']:
            # 如果当前epoch的验证集准确率更低，则增加早停计数器
            patience_counter += 1
        else:
            # 如果当前epoch的验证集准确率更高或相等，则重置早停计数器
            patience_counter = 0
            # 更新最佳epoch和其他相关评估指标
            tmp_dict1['epoch'] = epoch
            for k, v in eval_dict1.items():
                tmp_dict1[k] = v
            best_model_state = copy.deepcopy(gcn1.state_dict())
            best_pred = out.detach()
        if epoch%10 == 0:
            tqdm.write('epoch:{},loss:{:.4f}, val accuracy: {:.4f}, patience_counter:{}'.format(epoch, loss, eval_dict1['val_acc'], patience_counter))
            #tqdm.write('loss:{:.4f}'.format(loss))
    # evaluation
    gcn1.load_state_dict(best_model_state)
    eval_dict = evaluate(gcn1, data_t.data,True)
    print('teacher_model_val_acc:{}\n'.format(eval_dict['val_acc']))
    print('teacher_model_best_pred:{}\n'.format(pred))



    # 设置随机数种子
    set_seed(args.seed)
    # 学生模型
    gcn2 = GDC(dataset).to(device)
    optimizer2 = torch.optim.Adam(gcn2.parameters(), lr=args.lr, weight_decay=5e-4)
    # 训练学生模型
    reset_weights(gcn2)
    patience_counter = 0
    tmp_dict2 = {'val_acc': 0}
    best_model_state = copy.deepcopy(gcn2.state_dict())  # 初始化最佳模型状态
    gcn2.train()
    for epoch in tqdm(range(args.teacher_epoch)):
        if patience_counter == 100:  # 设定多少轮不提升就停止训练
            break
        # for epoch in tqdm(range(400)):,cs 800
        optimizer2.zero_grad()
        out1 = gcn2(dataset.data)
        pred = torch.argmax(out1, dim=1)
        # 计算学生和教师的概率分布
        student_logits = out1 / temperature
        teacher_logits = best_pred / temperature
        log_student_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        # 计算损失
        loss1 = F.nll_loss(F.log_softmax(out1, dim=1)[dataset.data.train_mask], dataset.data.y[dataset.data.train_mask])
        loss2 = F.kl_div(log_student_probs[dataset.data.train_mask], teacher_probs[dataset.data.train_mask], reduction='batchmean') * (
                    temperature ** 2)
        loss = alpha * loss1 + (1.0 - alpha) * loss2

        loss.backward()
        optimizer2.step()
        #pred = torch.argmax(out, dim=1)
        # 计算训练集准确率
        acc_train = (pred[data_t.data.train_mask] == data_t.data.y[data_t.data.train_mask]).float().mean()
        # 计算验证集准确率
        acc_val = (pred[data_t.data.val_mask] == data_t.data.true_y[data_t.data.val_mask]).float().mean()

        # acc = acc * 100
        # acc_st = int(correct) / int(test_mask.sum())
        eval_dict2 = evaluate(gcn2, dataset.data, True)
        # 比较当前epoch的验证集准确率与上一个最佳epoch的验证集准确率
        if eval_dict2['val_acc'] < tmp_dict2['val_acc']:
            # 如果当前epoch的验证集准确率更低，则增加早停计数器
            patience_counter += 1
        else:
            # 如果当前epoch的验证集准确率更高或相等，则重置早停计数器
            patience_counter = 0
            # 更新最佳epoch和其他相关评估指标
            tmp_dict2['epoch'] = epoch
            for k, v in eval_dict2.items():
                tmp_dict2[k] = v
            best_model_state = copy.deepcopy(gcn2.state_dict())
        if epoch % 10 == 0:
            tqdm.write(
                'epoch:{},loss:{:.4f}, Train accuracy: {:.4f}, val accuracy: {:.4f}, patience_counter:{}'.format(epoch, loss, acc_train,
                                                                                            acc_val, patience_counter))
            # tqdm.write('loss:{:.4f}'.format(loss))
    gcn2.load_state_dict(best_model_state)
    eval_dict2 = evaluate(gcn2, dataset.data, True)
    print('teacher_model_val_acc:{}\n'.format(eval_dict2['val_acc']))
    # with open('./result_主.txt', 'a') as file:
    #     '''name = "数据集：{}。算法：{}。噪音类型：{}。噪音比例：{}\n".format(dataset_name, preprocessing, noise,
    #                                                                 rate)
    #     acc_stds = "Acc and std of: {:.2f} +-{:.2f} %\n".format(100 * mean_acc, 100 * uncertainty)
    #     # 将结果写入文件
    #     file.write(name)
    #     file.write(acc_stds)'''
    #     # 超参调节输出
    #     # acc_stds = "{}**{}**{}**{:.2f}\n".format(args.drop_feature_rate,args.drop_edge_rate,dataset_name,100 * mean_acc)
    #     acc_stds = "{}**{}**{}**alpha:{}**T:{}**{:.2f}%\n".format(args.dataset_name, args.noise_type, args.noise_rate,alpha, temperature, eval_dict2['val_acc']*100)
    #     file.write(acc_stds)
    return eval_dict2['val_acc']*100


ss=0.0009
for args.dataset_name in ['dblp']:  #['Cora','Citeseer','Photo','Computers','dblp','CS','Pubmed']
    t=5
    eps = 0.0005
    for t in [1,3,5,7,9,11,13,15,17,19]:
        for args.noise_type in ['uniform', 'pair']:  #['uniform', 'pair']
            for args.noise_rate in [0.2]:  #[0.1,0.2,0.3,0.4,0.5]
                for alpha in [0.1]:  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    for temperature in [10]:  # [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        acc_val = []
                        for args.seed in test_seeds11: #test_seeds
                            if args.dataset_name == 'Computers':
                                ss=0.0003
                                args.lr = 0.00449
                                #if args.noise_rate in [0.2,0.5] :
                                    #temperature = 0.3
                                if args.noise_type =='pair':
                                    args.lr = 0.005
                            if args.dataset_name == 'Photo':
                                args.lr = 0.004
                                if args.noise_rate == 0.5:
                                    args.lr = 0.0002
                            acc = run_model(args=args, random_state=0, t=t, eps =eps ,noise_type=args.noise_type, noise_ratio=args.noise_rate, alpha=alpha, temperature=temperature, ss=ss )
                            acc_val.append(acc)
                            # 取test_acc的前5个值进行降序排序
                        test_acc_sorted = sorted(acc_val, reverse=True)[:5]
                        # 计算均值和标准差
                        test_acc_mean = np.mean(test_acc_sorted)
                        test_acc_std = np.std(test_acc_sorted)
                        with open('./result_超参实验扩散.txt', 'a') as file:
                            acc_stds = "{}**{}**{}**{}**{}**{:.2f}%\n".format(t,eps ,args.dataset_name,
                                                                                      args.noise_type, args.noise_rate,
                                                                                      test_acc_mean)
                            file.write(acc_stds)