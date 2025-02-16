__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

import os.path as osp

import numpy as np
from scipy.linalg import expm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull,WikiCS

from seeds import development_seed
from numpy.testing import assert_array_almost_equal
DATA_PATH = 'data'
# def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
#     # path = os.path.join(DATA_PATH, name)
#     if name in ['Cora', 'Citeseer', 'Pubmed']:
#         # data = Planetoid(path, name)
#         path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')  # or 'data/'
#         # print(osp.dirname(osp.realpath(__file__)),path)
#         data = Planetoid(path, name)
#     elif name in ['Computers', 'Photo']:
#         path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
#         data = Amazon(path, name)
#     # elif name == 'CoauthorCS':
#     #     data = Coauthor(path, 'cs')
#     else:
#         raise Exception('Unknown data.')
#
#     if use_lcc:
#         lcc = get_largest_connected_component(data)
#
#         x_new = data.data.x[lcc]
#         y_new = data.data.y[lcc]
#
#         row, col = data.data.edge_index.numpy()
#         edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
#         edges = remap_edges(edges, get_node_mapper(lcc))
#
#         data = Data(
#             x=x_new,
#             edge_index=torch.LongTensor(edges),
#             y=y_new,
#             train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
#             test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
#             val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
#         )
#         data.data = data
#
#     return data



def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)  #用于从给定的起始节点开始，找到与其连通的所有节点。
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes, dtype=np.float32)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes, dtype=np.float32) - H.astype(np.float32)))



def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:  #在这里修改，测试集每个类别选取几个节点
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

def uniform_trans(n_class, noise_ratio):
    # uniform transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = np.float64(noise_ratio) / np.float64(n_class - 1) * np.ones((n_class,n_class))
    np.fill_diagonal(trans,(np.float64(1) - np.float64(noise_ratio))*np.ones(n_class))
    # print(trans.sum(0))
    diag_idx = np.arange(n_class)
    # make sure that sum of every row is 0
    trans[diag_idx,diag_idx] = trans[diag_idx,diag_idx] + 1.0 - trans.sum(0)
    # assert_array_almost_equal(a,b,decimal=6)
    assert_array_almost_equal(trans.sum(axis=1),1,1)
    # print(diag_idx)
    return trans #这里返回的是一个转换矩阵，大小为[n_class,n_class],n_class为类别数量

def pair_trans(n_class, noise_ratio):
    # pair_transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = (1.0 - np.float64(noise_ratio)) * np.eye(n_class)
    for i in range(n_class):
        trans[i-1,i] = np.float64(noise_ratio)
    assert_array_almost_equal(trans.sum(axis=1), 1, 1)
    return  trans  #这里同样返回了一个转换矩阵。

def inter_class_noisify(labels, trans,random_state=0):#对标签进行噪音化处理，生成新的标签
    # flip classes according to transition matrix
    assert trans.shape[0] == trans.shape[1]
    assert np.max(labels) < trans.shape[0]
    # assert torch.max(labels) < trans.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(trans.sum(axis=1), np.ones(trans.shape[1]))   #断言确保转换矩阵是行随机矩阵，即每行的和为1。
    assert (trans >= 0.0).all() #确保非负

    # trans = torch.from_numpy(trans)
    m = labels.shape[0] #零维大小，一共有多少个，标签数组是一维的
    new_labels = labels.copy()
    # new_labels = torch.clone(labels)
    # random number generator
    flipper = np.random.RandomState(random_state)#创建一个随机数生成器

    for idx in np.arange(m):
        # i ranges from 0 to n_class-1
        i = labels[idx]
        flipped = flipper.multinomial(1, trans[i, :], 1)[0]#因为trans是一个方阵，比如[第一行,的所有列]代表第一类
        # sample_label = torch.multinomial(trans[i,:],1)[0]
        new_labels[idx] = np.where(flipped == 1)[0]
        # new_labels[idx] = sample_label

    return new_labels

def noisify_p(labels,n_class,noise_ratio,random_state=None,noise_type='pair'):#对输入的标签数组进行噪声处理，集成了上边三个函数
    if noise_ratio > 0.0:# 判断噪声比率是否大于0，如果大于0，则进行噪声处理，否则直接返回原始标签
        if noise_type == 'uniform':
            print("Uniform noise")
            trans = uniform_trans(n_class,noise_ratio)
        elif noise_type == 'pair':
            print("Pair noise")
            trans = pair_trans(n_class,noise_ratio)
        else:
            print("Noise type not implemented")

        noisy_labels = inter_class_noisify(labels,trans,random_state)
        actual_noise_ratio = (noisy_labels != labels).mean()# 计算实际噪声比率，即噪声标签与原始标签不同的比例
        assert actual_noise_ratio > 0.0# 断言确保实际噪声比率大于0，即至少有一些标签被改变了
        #print("Actual noise ratio:{:.2f}".format(actual_noise_ratio))
        print("实际噪声比率:{:.2f}".format(actual_noise_ratio))
        labels = noisy_labels
    else:
        #print("Actual noise ratio:0")
        print("实际噪声比率:0")
        trans = np.eye(n_class)

    return labels,trans

def get_dataset(name: str, use_lcc: bool = True,noise: str='uniform',rate: float=0.2) -> InMemoryDataset:
    # path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        #data = Planetoid(path, name)
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')  # or 'data/'
        # print(osp.dirname(osp.realpath(__file__)),path)
        dataset = Planetoid(path, name)
    elif name in ['Computers', 'Photo']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Amazon(path, name)
    # elif name == 'CoauthorCS':
    #     data = Coauthor(path, 'cs')
    elif  name in ['dblp']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = CitationFull(path, name)
    elif name in ['CS']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/cs')
        # data = WikiCS(path, args.dataset_name)
        dataset = WikiCS(path)
    else:
        raise Exception('Unknown data.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]      #获取最大通分的特征信息
        y_new = dataset.data.y[lcc]      #获取最大通分的标签信息
        print('正常y！！！！！！！！！！')
        print(y_new)

        row, col = dataset.data.edge_index.numpy()   #获取最大通分边的行列节点
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]    #eg：row = [0, 0, 1, 2, 3, 4]    col = [1, 2, 2, 3, 4, 0]-》edges = [[0, 1], [0, 2], [1, 2], [2, 3]]
        edges = remap_edges(edges, get_node_mapper(lcc))  #重映射的目的是将节点索引转换为从 0 开始的连续整数索引

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

        data = dataset
        #noise_type = 'uniform'
        noise_type = noise
        print("注意！！！！！！！！！！这是噪音类型")
        print(noise)
        random_state = 0
        #noise_ratio = 0.1
        noise_ratio=rate
        print("注意！！！！！！！！！！这是噪音比率")
        print(rate)
        device = 'cuda:0'
        #data = get_dataset(name='Cora', use_lcc=True)
        # data = data.to(device)
        label = (data.y).clone().detach().to(device).squeeze()
        print("label", label)
        n_features = data.num_features
        num_classes = int(label.max() - label.min()) + 1
        # n_nodes = data.num_nodes
        # num_classes = data.num_classes
        noisy_label, trans = noisify_p(label.cpu().numpy().flatten(), n_class=num_classes, noise_type=noise_type,
                                       random_state=random_state, noise_ratio=noise_ratio)
        noisy_label = torch.from_numpy(noisy_label)
        # noisy_label = noisy_label.to(device)
        print('噪音标签！！！！！！！！！！！！！！1')
        print(noisy_label)

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=noisy_label,
            true_y=y_new,
            train_mask=torch.zeros(noisy_label.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(noisy_label.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(noisy_label.size()[0], dtype=torch.bool)
        )
        dataset.data = data


    return dataset

class HeatDataset(InMemoryDataset):
    """
    Dataset preprocessed with GDC using heat kernel diffusion.
    Note that this implementations is not scalable
    since we directly calculate the matrix exponential
    of the adjacency matrix.
    """

    def __init__(self,
                 name: str = '',
                 use_lcc: bool = True,
                 noise: str='',
                 rate: float=0.2,
                 t: float = 5.0,
                 k: int = 16,
                 eps: float = 0.0000):
        self.noise=noise
        self.rate=rate
        self.name = name
        self.use_lcc = use_lcc

        self.t = t
        self.k = k
        self.eps = eps

        super(HeatDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        base = get_dataset(name=self.name, use_lcc=self.use_lcc,noise=self.noise,rate=self.rate)
        # generate adjacency matrix from sparse representation    ！！！！！！！！由稀疏表示生成邻接矩阵！！！！！
        adj_matrix = get_adj_matrix(base)
        # get heat matrix as described in Berberidis et al., 2019 ！！！！！！！！#如Berberidis等人所述获得热矩阵:！！！！！！
        heat_matrix = get_heat_matrix(adj_matrix,
                                      t=self.t)
        # if self.k !=0:
        #     print(f'Selecting top {self.k} edges per node.')
        #     heat_matrix = get_top_k_matrix(heat_matrix, k=self.k)
        # elif self.eps !=0:
        #     print(f'Selecting edges with weight greater than {self.eps}.')
        #     heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
        # else:
        #     raise ValueError
        heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(heat_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(heat_matrix[i, j])  # !!!!!!!!!!!!!!!!!!!!!!!!!!这就是裁剪之后的热矩阵！！！！！！！！！！！！！
        edge_index = [edges_i, edges_j]  # ！！这是边的连接关系，发生了变化

        data = Data(  # 在新的数据对象中，
            x=base.data.x,  # 无变化
            edge_index=torch.LongTensor(edge_index),  # 变化
            edge_attr=torch.FloatTensor(edge_attr),  # 新加入
            y=base.data.y,  # 变化
            true_y=base.data.true_y,
            train_mask=torch.zeros(base.data.train_mask.size()[0], dtype=torch.bool),  # 无变化
            test_mask=torch.zeros(base.data.test_mask.size()[0], dtype=torch.bool),  # 无变化
            val_mask=torch.zeros(base.data.val_mask.size()[0], dtype=torch.bool)  # 无变化
        )
        print('测试热核扩散对象')
        print(data)
        # 对于edge_attr列表随后被填充为裁剪后热矩阵中的相应元素值，这些值现在代表了图中边的权重。这些权重可以被视为节点之间连接强度的度量，对于图神经网络或其他图算法来说是非常重要的信息。
        # 因此，edge_attr属性确实可以表示边的权重，在这个上下文中，这些权重是从热矩阵中派生出来的，并且经过了基于权重的裁剪过程。这些权重对于理解和分析图的结构以及训练基于图的机器学习模型都是非常有用的特征。
        data, slices = self.collate([data])

        print(data)

        torch.save((data, slices), self.processed_paths[0])

    def __str__(self) -> str:
        return f'{self.name}_heat_t={self.t}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}_rate={self.rate}_noise={self.noise}'
import  os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# a=HeatDataset();
# print('*******************************************')
# #a.process()
# print(a.data)
# print('*******************************************')
# #get_dataset('cora')





# def get_dataset2():
#
#     noise_type = 'uniform'
#     random_state = 0
#     noise_ratio = 0.1
#     device = 'cpu'
#     data = get_dataset(name='Cora', use_lcc=True)
#     # data = data.to(device)
#     label = (data.y).clone().detach().to(device).squeeze()
#     print("label", label)
#     n_features = data.num_features
#     num_classes = int(label.max() - label.min()) + 1
#     # n_nodes = data.num_nodes
#     # num_classes = data.num_classes
#     noisy_label, trans = noisify_p(label.cpu().numpy().flatten(), n_class=num_classes, noise_type=noise_type,
#                                    random_state=random_state, noise_ratio=noise_ratio)
#     # noisy_label = torch.from_numpy(noisy_label)
#     # noisy_label = noisy_label.to(device)
#     print(noisy_label)
