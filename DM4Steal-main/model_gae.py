
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import  APPNP
import numpy as np
import scipy.sparse as sp
import os

from sklearn.metrics import roc_auc_score, average_precision_score, auc, roc_curve



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj





class AE(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(AE, self).__init__()
        self.l1 = nn.Linear (input_feat_dim,hidden_dim1)
        self.l2 = nn.Linear (hidden_dim1,hidden_dim2)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self,x):
        hidden1 = self.l1(x)
        z=self.l2(hidden1)
        return self.dc(z),z

def loss_function_GNAE(preds, labels, norm, pos_weight):

    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    return cost

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_train =int (np.floor(edges.shape[0] / 5))
    num_test = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    train_edge_idx = all_edge_idx[:num_train]
    test_edge_idx = all_edge_idx[num_train:(num_test+num_train)]

    test_edges = edges[test_edge_idx]
    train_edges = edges[train_edge_idx]


    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)#与
        return np.any(rows_close)#或



    test_edges_false = []
    while len(test_edges_false) < (len(test_edges)):
        idx_i = np.random.randint(0, adj.shape[0]-2)
        idx_j = np.random.randint(0, adj.shape[0]-2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])



    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, test_edges, test_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)  # 相乘得到重构邻接矩阵,class 'numpy.ndarray'
    preds = []
    pos = []
    false_right_node0 = []
    false_right_node1  = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
        if sigmoid(adj_rec[e[0], e[1]])>0.5:
            false_right_node0.append(e[0])
            false_right_node1.append(e[1])
    # 这里获得的preds列表中的值不是整数
    preds_neg = []
    neg = []
    right_false_node0 = []
    right_false_node1 = []
    for e in edges_neg:
        neg.append(adj_orig[e[0], e[1]])
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        # if sigmoid(adj_rec[e[0], e[1]]) < 0.5:
        #     right_false_node0.append(e[0])
        #     right_false_node1.append(e[1])

    preds_all = np.hstack([preds, preds_neg])  # 在水平方向上平铺拼接数组
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    '''fpr, tpr, _ = roc_curve(labels_all, preds_all, sample_weight=None)
    auc_score = auc(fpr, tpr)roc_score'''

    return roc_score, ap_score,false_right_node0, false_right_node1#, auc_score

