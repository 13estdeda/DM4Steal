import torch
import logging
import time
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from models.gcn import GCN, embedding_GCN
from topology_attack import PGDAttack
from utils import *
from dataset import Dataset
import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score,roc_auc_score
import scipy.io as sio
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from easydict import EasyDict as edict
import torch_geometric.transforms as T
import pickle as pkl
import networkx as nx
import sys
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from evaluation.stats import eval_torch_batch

from model.langevin_mc import LangevinMCSampler
from sample import sample_main
from util.arg_helper import edict2dict, parse_arguments, get_config, process_config, set_seed_and_logger, load_data
from util.graph_utils import gen_list_of_data
from util.loading_utils import get_mc_sampler, get_score_model, eval_sample_batch
from util.visual_utils import plot_graphs_adj
from models.gat import GAT, embedding_gat
from models.graphsage import graphsage,embedding_graphsage

def pingjie(adj,infer_adj,sample_adj,batch):

    list1 = []
    list2 = []
    for i in range(adj.shape[0]):
        if sum(adj[i]) > 1 and sum(adj[i]) < 16:
            list1.append(i)
        else:
            list2.append(i)
    for i in range(batch):
        sample_list = []
        sample_batch = adj[list1[i]]
        for j in range(sample_batch.shape[0]):
            if sample_batch[j] == 1:
                sample_list.append(j)
        for j in range(len(sample_list)):
            infer_adj[sample_list[j]][list1[i]] = infer_adj[list1[i]][sample_list[j]] = sample_adj[i][j+1][0]
            for k in range(len(sample_list)):
                infer_adj[sample_list[j]][sample_list[k]] =infer_adj[sample_list[k]][sample_list[j]]= sample_adj[i][j+1][k+1]

    infer_adj[infer_adj>1]=1
    infer_adj[infer_adj<0]=0
    return infer_adj

def find_attcak_index(adj,rate):
    attack_index=[]
    list1=[]
    for i in range(adj.shape[0]):
        if sum(adj[i]) > 1 and sum(adj[i]) < 16:
            list1.append(i)
    for i in range(len(list1)):
        sample_batch = adj[list1[i]]
        if list1[i] not in attack_index:
            attack_index.append(list1[i])
        for j in range (sample_batch.shape[0]):
            if sample_batch[j]==1 and j not in attack_index:
                attack_index.append(j)
        if len(attack_index)>int(adj.shape[0]*rate):
            break
    attack_index=attack_index[:int(adj.shape[0]*rate)]
    attack_index=np.array(attack_index)
    return attack_index

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_original(datapath_str, dataset_str):
    """
    Loads input data from gcn/data/dataset directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, names[i]),
                    'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(
        datapath_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder),
            max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[
        test_idx_range, :]  # order the test features
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[
        test_idx_range, :]  # order the test labels


    all_id_list = list(range(len(labels)))
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list

    # original
    train_mask = sample_mask(idx_train, labels.shape[0])  # index =1, others = 0
    val_mask = sample_mask(idx_val, labels.shape[0])  # index =1, others = 0
    test_mask = sample_mask(idx_test, labels.shape[0])  # index =1, others = 0



    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[
        train_mask, :]  # only the mask position has the true label, others are set to 0
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_tu(datapath_str, dataset_str):
    print("load_data_tu: %s"%dataset_str)
    names = ['attr', 'graph']
    objects = []
    for i in range(len(names)):
        with open(
                "dataset/{}/{}_{}.pkl".format(dataset_str, dataset_str, names[i]),
                'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    attr, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = np.array([attr[i]["feature_vec"] for i in range(len(attr))])
    labels = np.array([attr[i]["label"] for i in range(len(attr))])

    all_id_list = list(range(len(labels)))
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(datapath_str, dataset_str):
    if dataset_str in ["citeseer", "cora", "pubmed"]:
        return load_data_original(datapath_str, dataset_str)
    elif dataset_str in ["AIDS", "COX2", "DHFR", "ENZYMES", "PROTEINS_full"]:
        return load_data_tu(datapath_str, dataset_str)
    else:
        raise Exception("Invalid dataset!", dataset_str)


def sample(adj,features,batch):
    z = np.sum(adj.numpy(), axis=1)
    list1=[]
    list2=[]
    for i in range(adj.shape[0]):
        if sum(adj[i]) > 1 and sum(adj[i])<16:
            list1.append(i)
        else:
            list2.append(i)
        if len(list1)>batch:
            print("find subgraph")
            break
    init_adj=np.zeros((batch,16,16))
    init_feature=np.zeros((batch,16,features.shape[1]))
    for i in range(batch):
        sample_list = []
        sample_batch=adj[list1[i]]
        init_feature[i][0]=features[list1[i]]
        t=time.time()
        for j in range(sample_batch.shape[0]):
            if sample_batch[j] == 1:
                sample_list.append(j)
        print("shijian:",time.time()-t)
        t=time.time()
        for j in range(len(sample_list)):
            init_adj[i][0][j+1]=init_adj[i][j+1][0]=1
            for k in range(len(sample_list)):
                if adj[sample_list[j]][sample_list[k]] == 1:
                    init_adj[i][j + 1][k + 1] =init_adj[i][k + 1][j + 1] = 1
                    init_feature[i][j+1]=features[sample_list[j]]
        print("shijian2:",time.time()-t)
    sample_adj=torch.from_numpy(init_adj)
    sample_features=torch.from_numpy(init_feature)
    return sample_adj,sample_features,list1

def sample_test(adj,features,batch):

    list1=[]
    list2=[]
    for i in range(adj.shape[0]):
        if sum(adj[i]) > 1 and sum(adj[i])<16:
            list1.append(i)
        else:
            list2.append(i)
    init_adj=np.zeros((batch,16,16))
    init_feature=np.zeros((batch,16,features.shape[1]))
    for i in range(batch,2*batch):
        sample_list = []
        sample_batch=adj[list1[i]]
        init_feature[i-batch][0]=features[list1[i]]
        for j in range(sample_batch.shape[0]):
            if sample_batch[j] == 1:
                sample_list.append(j)
        for j in range(len(sample_list)):
            init_adj[i-batch][0][j + 1] = init_adj[i-batch][j + 1][0] = 1
            for k in range(len(sample_list)):
                if adj[sample_list[j]][sample_list[k]] == 1:
                    init_adj[i-batch][j + 1][k + 1] =init_adj[i-batch][k + 1][j + 1] = 1
                    init_feature[i-batch][j+1]=features[sample_list[j]]
    sample_adj=torch.from_numpy(init_adj)
    sample_features=torch.from_numpy(init_feature)
    return sample_adj,sample_features

def sample_infer(adj,features,infer_adj,batch):
    # z = np.sum(adj.numpy(), axis=1)
    infer=infer_adj.numpy()
    list1 = []
    list2 = []

    for i in range(adj.shape[0]):
        if sum(adj[i]) > 1 and sum(adj[i]) < 16:
            list1.append(i)
        else:
            list2.append(i)
    init_adj = np.zeros((batch, 16, 16))
    init_feature = np.zeros((batch, 16, features.shape[1]))
    print('list:',len(list1))
    for i in range(batch,2*batch):

        sample_list = []
        sample_batch=adj[list1[i]]
        init_feature[i-batch][0]=features[list1[i]]
        t=time.time()
        for j in range(sample_batch.shape[0]):
            if sample_batch[j] == 1:
                sample_list.append(j)
        print("shijian:",time.time()-t)
        t=time.time()
        for j in range(len(sample_list)):
            init_adj[i-batch][0][j+1]=init_adj[i-batch][j+1][0]=infer[list1[i]][sample_list[j]]
            for k in range(len(sample_list)):
                init_adj[i-batch][j + 1][k + 1] = infer[sample_list[j]][sample_list[k]]
            init_feature[i-batch][j+1]=features[sample_list[j]]
        print("shijian2:",time.time()-t)
    sample_adj=torch.from_numpy(init_adj)
    sample_features=torch.from_numpy(init_feature)
    return sample_adj,sample_features

def loss_func(score_list, grad_log_q_noise_list, sigma_list):
    loss = 0.0
    loss_items = []
    for score, grad_log_q_noise, sigma in zip(score_list, grad_log_q_noise_list, sigma_list):
        cur_loss = 0.5 * sigma ** 2 * ((score - grad_log_q_noise) ** 2).sum(dim=[-1, -2]).mean()
        loss_items.append(cur_loss.detach().cpu().item())
        loss = loss + cur_loss
    assert isinstance(loss, torch.Tensor)
    return loss, loss_items


def fit(model, optimizer, mcmc_sampler, train_adj,train_x ,test_adj,test_x, max_node_number, max_epoch=20, config=None,
        save_interval=50,
        sample_interval=1,
        sigma_list=None,
        sample_from_sigma_delta=0.0
        ):
    logging.info(f"{sigma_list}, {sample_from_sigma_delta}")
    assert isinstance(mcmc_sampler, LangevinMCSampler)

    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    for epoch in range(max_epoch):
        train_losses = []
        train_loss_items = []
        test_losses = []
        test_loss_items = []
        t_start = time.time()
        model.train()
        train_adj_b=train_adj
        train_x_b =train_x

            # here,
            # train_adj_b is of size [batch_size, N, N]
            # train_x_b is of size [batch_size, N, F_i]
        train_adj_b = train_adj_b.to(config.dev)
        train_x_b = train_x_b.to(config.dev)

        train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        if isinstance(sigma_list, float):
                sigma_list = [sigma_list]
        train_x_b, train_noise_adj_b, \
        train_node_flag_b, grad_log_q_noise_list = \
            gen_list_of_data(train_x_b, train_adj_b,
                                 train_node_flag_b, sigma_list)
            # thereafter,
            # train_noise_adj_b is of size [len(sigma_list) * batch_size, N, N]
            # train_x_b is of size [len(sigma_list) * batch_size, N, F_i]
        optimizer.zero_grad()
        score = model(x=train_x_b,
                          adjs=train_noise_adj_b,
                          node_flags=train_node_flag_b)

        loss, loss_items = loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
        train_loss_items.append(loss_items)
        loss.backward()

        optimizer.step()
        train_losses.append(loss.detach().cpu().item())
        scheduler.step(epoch)
        assert isinstance(model, nn.Module)
        model.eval()

        test_adj_b=test_adj
        test_x_b =test_x
        test_adj_b = test_adj_b.to(config.dev)
        test_x_b = test_x_b.to(config.dev)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
            gen_list_of_data(test_x_b, test_adj_b,
                                 test_node_flag_b, sigma_list)
        with torch.no_grad():
            score = model(x=test_x_b, adjs=test_noise_adj_b,
                              node_flags=test_node_flag_b)
        loss, loss_items = loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
        test_loss_items.append(loss_items)
        test_losses.append(loss.detach().cpu().item())

        mean_train_loss = np.mean(train_losses)
        mean_test_loss = np.mean(test_losses)
        mean_train_loss_item = np.mean(train_loss_items, axis=0)
        mean_train_loss_item_str = np.array2string(mean_train_loss_item, precision=2, separator="\t", prefix="\t")
        mean_test_loss_item = np.mean(test_loss_items, axis=0)
        mean_test_loss_item_str = np.array2string(mean_test_loss_item, precision=2, separator="\t", prefix="\t")

        logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                     f'train loss: {mean_train_loss:+.3e} | '
                     f'test loss: {mean_test_loss:+.3e} | ')

        logging.info(f'epoch: {epoch:03d}| '
                     f'train loss i: {mean_train_loss_item_str} '
                     f'test loss i: {mean_test_loss_item_str} | ')

        if epoch % save_interval == save_interval - 1:
            to_save = {
                'model': model.state_dict(),
                'sigma_list': sigma_list,
                'config': edict2dict(config),
                'epoch': epoch,
                'train_loss': mean_train_loss,
                'test_loss': mean_test_loss,
                'train_loss_item': mean_train_loss_item,
                'test_loss_item': mean_test_loss_item,
            }
            torch.save(to_save, os.path.join(config.model_save_dir,
                                             f"{config.dataset.name}_{sigma_list}.pth"))
            # torch.save(to_save, os.path.join(config.save_dir, "model.pth"))

        if epoch % sample_interval == sample_interval - 1:
            model.eval()
            test_adj_b, test_x_b = test_adj,test_x
            test_adj_b = test_adj_b.to(config.dev)
            test_x_b = test_x_b.to(config.dev)
            if isinstance(config.mcmc.grad_step_size, (list, tuple)):
                grad_step_size = config.mcmc.grad_step_size[0]
            else:
                grad_step_size = config.mcmc.grad_step_size
            step_size = grad_step_size * \
                        torch.tensor(sigma_list).to(test_x_b) \
                            .repeat_interleave(test_adj_b.size(0), dim=0)[..., None, None] ** 2
            test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
            test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data(test_x_b, test_adj_b,
                                 test_node_flag_b, sigma_list)
            init_adjs = test_noise_adj_b
            with torch.no_grad():
                sample_b, _ = mcmc_sampler.sample(config.sample.batch_size,
                                                  lambda x, y: model(test_x_b, x, y),
                                                  max_node_num=max_node_number, step_num=None,
                                                  init_adjs=init_adjs, init_flags=test_node_flag_b,
                                                  is_final=True,
                                                  step_size=step_size)
                sample_b_list = sample_b.chunk(len(sigma_list), dim=0)
                init_adjs_list = init_adjs.chunk(len(sigma_list), dim=0)
                for sigma, sample_b, init_adjs in zip(sigma_list, sample_b_list, init_adjs_list):
                    sample_from_sigma = sigma + sample_from_sigma_delta
                    eval_sample_batch(sample_b, mcmc_sampler.end_sample(test_adj_b, to_int=True)[0], init_adjs,
                                      config.save_dir, title=f'epoch_{epoch}_{sample_from_sigma}.pdf')

                    # if init_adjs is not None:
                    #     plot_graphs_adj(mcmc_sampler.end_sample(init_adjs, to_int=True)[0],
                    #                     node_num=test_node_flag_b.sum(-1).cpu().numpy(),
                    #                     title=f'epoch_{epoch}_{sample_from_sigma}_init.pdf',
                    #                     save_dir=config.save_dir)
                    # result_dict = eval_torch_batch(mcmc_sampler.end_sample(test_adj_b, to_int=True)[0],
                    #                                sample_b, methods=None)
                    # logging.info(f'MMD {epoch} {sample_from_sigma}: {result_dict}')
    return  sample_b


def train_main(config, args,train_adj,train_x,test_adj,test_x):
    set_seed_and_logger(config, args)

    mc_sampler = get_mc_sampler(config)

    # here, the `model` get `num_classes=len(sigma_list)`
    model = get_score_model(config)

    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)
    logging.info(f"Parameters: \n{param_string}")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters Count: {total_params}, Trainable: {total_trainable_params}")
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.lr_init,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.train.weight_decay)
    sample_b=fit(model, optimizer, mc_sampler, train_adj,train_x,test_adj,test_x,
        max_node_number=config.dataset.max_node_num,
        max_epoch=config.train.max_epoch,
        config=config,
        save_interval=config.train.save_interval,
        sample_interval=config.train.sample_interval,
        sigma_list=config.train.sigmas,
        sample_from_sigma_delta=0.0)

    return sample_b

    # sample_main(config, args)

def dtet(adj, features, labels, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return output.detach()

def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj

def preprocess_Adj(adj, feature_adj):
    n=len(adj)
    cnt=0
    adj=adj.numpy()
    feature_adj=feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
                adj[i][j]=1.0
                cnt+=1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def metric(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP: %f" % (auc(fpr, tpr), average_precision_score(real_edge, pred_edge)))

def metric_2(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    # index = np.where(real_edge == 0)[0]
    # index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    # real_edge = np.delete(real_edge, index_delete)
    # pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f" % (auc(fpr, tpr)))

def metric_3(ori_adj, inference_adj, idx):

    real_edge=ori_adj.reshape(-1)
    pred_edge=inference_adj.reshape(-1)
    real_edge_list_pos=[]
    real_edge_list_neg=[]
    pred_edge_list_pos=[]
    pred_edge_list_neg=[]
    for i in range(real_edge.shape[0]):
        if real_edge[i]>0.5:
            real_edge_list_pos.append(real_edge[i])
            pred_edge_list_pos.append(pred_edge[i])
        else:
            real_edge_list_neg.append(real_edge[i])
            pred_edge_list_neg.append(pred_edge[i])
    real_all=np.hstack([real_edge_list_pos,real_edge_list_neg])
    pred_all=np.hstack([pred_edge_list_pos,pred_edge_list_neg])
    pred_all[pred_all<0]=0
    pred_all[pred_all>1]=1

    AUC=roc_auc_score(real_all,pred_all)
    print("Inference attack AUC: %f" % (AUC))

def label_onehot_to_lebel_valve(onehot):
    label_number=onehot.shape[1]
    label=np.zeros((label_number,1))
    for i in range(label_number):
        label[i][0]=i

    label=np.dot(onehot,label)
    return label


def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
                #pred_edge.append(np.dot(output[idx[i]], output[idx[j]])/(np.linalg.norm(output[idx[i]])*np.linalg.norm(output[idx[j]])))
                #pred_edge.append(-np.linalg.norm(output[idx[i]]-output[idx[j]]))
                #pred_edge.append(np.dot(features[idx[i]], features[idx[j]]) / (np.linalg.norm(features[idx[i]]) * np.linalg.norm(features[idx[j]])))

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'COX2', 'ENZYMES','PROTEINS_full'], help='dataset')
parser.add_argument('--shadowdataset', type=str, default='pubmed',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'COX2', 'ENZYMES','PROTEINS_full'], help='dataset')
parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)
parser.add_argument('--datapath', type=str, default="dataset/", help="data path")
parser.add_argument('--defense', type=str, default="DP_gauss", choices=['DP_gauss', 'DP_laplace', 'noise', 'droup_out'],help="defense methods")
args = parser.parse_args()
# args, unknown = parser.parse_known_args()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.datapath, args.dataset)

features=sp.csr_matrix(features)
labels=label_onehot_to_lebel_valve(y_test)
labels=labels.reshape(-1).astype(int)
init_adj=sp.csr_matrix(np.zeros((adj.shape[0],adj.shape[1])))

idx_train,idx_val,idx_test=get_train_val_test_gcn_2(labels,args.seed)
#choose the target nodes
idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))
num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_attack)**2)

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)

adj_shadow, features_shadow, y_train_shadow, y_val_shadow, y_test_shadow, train_mask_shadow, val_mask_shadow, test_mask_shadow = load_data(args.datapath, args.shadowdataset)

features_shadow=sp.csr_matrix(features_shadow)
labels_shadow=label_onehot_to_lebel_valve(y_test_shadow)
labels_shadow=labels_shadow.reshape(-1).astype(int)
init_adj_shadow=sp.csr_matrix(np.zeros((adj_shadow.shape[0],adj_shadow.shape[1])))

idx_train_shadow,idx_val_shadow,idx_test_shadow=get_train_val_test_gcn_2(labels_shadow,args.seed)
#choose the target nodes
idx_attack_shadow = np.array(random.sample(range(adj_shadow.shape[0]), int(adj_shadow.shape[0]*args.nlabel)))
num_edges_shadow = int(0.5 * args.density * adj_shadow.sum()/adj_shadow.shape[0]**2 * len(idx_attack_shadow)**2)

adj_shadow, features_shadow, labels_shadow = preprocess(adj_shadow, features_shadow, labels_shadow, preprocess_adj=False, onehot_feature=False)

# idx_attack=find_attcak_index(adj,args.nlabel)
# num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_attack)**2)

# to tensor
feature_adj = dot_product_decode(features)
#preprocess_adj = preprocess_Adj(adj, feature_adj)
init_adj = torch.FloatTensor(init_adj.todense())
# initial adj is set to zero matrix

# Setup Victim Model

victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                   dropout=0.5, weight_decay=5e-4,defense=args.defense, device=device)
# victim_model = GAT(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
#                    dropout=0.5, alpha=0.2,nheads=4, device=device)
# victim_model = graphsage(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train, idx_val)

features_vic, adj_vic,_ = to_tensor(features, adj, labels, device=device)
embed=victim_model(features_vic,adj_vic)

embedding = embedding_GCN(nfeat=features.shape[1], nhid=16,defense=args.defense, device=device)
# embedding = embedding_gat(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
#                    dropout=0.5, alpha=0.2,nheads=4, device=device)
# embedding=embedding_graphsage(nfeat=features.shape[1], nhid=16, device=device)
# embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))

victim_model_shadow = GCN(nfeat=features_shadow.shape[1], nclass=labels_shadow.max().item() + 1, nhid=16,
                   dropout=0.5, weight_decay=5e-4,defense=args.defense, device=device)
# victim_model_shadow = GAT(nfeat=features_shadow.shape[1], nclass=labels_shadow.max().item() + 1, nhid=16,
#                    dropout=0.5, alpha=0.2,nheads=4, device=device)
# victim_model_shadow = graphsage(nfeat=features_shadow.shape[1], nhid=16, nclass=labels_shadow.max().item() + 1, dropout=0.5, device=device)
#
victim_model_shadow = victim_model_shadow.to(device)
victim_model_shadow.fit(features_shadow, adj_shadow, labels_shadow, idx_train_shadow, idx_val_shadow)

features_vic_shadow, adj_vic_shadow,_ = to_tensor(features_shadow, adj_shadow, labels_shadow, device=device)
embed_shadow=victim_model_shadow(features_vic_shadow,adj_vic_shadow)

# Setup Attack Model

model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)


def main():
    # list=[]
    # list_2=[]
    # for i in range(adj.shape[0]):
    #     if sum(adj[i])>4:
    #
    #         list.append(i)
    #     else:
    #         list_2.append(i)
    # for_test=np.zeros((2708,2708))
    # for_test[list]=1
    # z_1=np.delete(for_test,list_2,axis=0)
    # print("ok")
    # adj_test=np.multiply(adj.numpy(),for_test)
    # # adj_test=np.delete(adj_test,list_2,axis=0)
    # idx_t=[]
    # for i in range (idx_attack.shape[0]):
    #     if idx_attack[i] not in list_2:
    #         idx_t.append(idx_attack[i])
    # idx_t=np.array(idx_t)
    #
    # z=np.sum(adj.numpy(),axis=1)
    # z=np.sum(z)
    # sample=adj[38].numpy()
    # sample_array=np.zeros((9,9))
    # sample_array[0]=1
    # sample_array[0][0]=0
    #
    # sample_list=[]
    # for i in range (sample.shape[0]):
    #     if sample[i]==1:
    #         sample_list.append(i)
    # for i in range (len(sample_list)):
    #     for j in range(len(sample_list)):
    #         if adj[sample_list[i]][sample_list[j]]==1:
    #             sample_array[i+1][j+1]=1
    # for i in range(len(sample_list)):
    #     sample_array[0][i+1]=1


    if features_shadow.cpu().detach().shape[1] > features.cpu().detach().shape[1]:
        zero_=np.zeros(shape=(features.shape[0],(features_shadow.shape[1]-features.shape[1])))
        features_d=np.array(features.cpu().detach())
        features_d=np.hstack((features_d,zero_))
        features_d = torch.from_numpy(features_d)
        features_shadow_d=features_shadow.cpu().detach()


    else:
        zero_ = np.zeros(shape=(features_shadow.shape[0], (features.shape[1] - features_shadow.shape[1])))
        features_shadow_d = np.array(features_shadow.cpu().detach())
        features_shadow_d = np.hstack((features_shadow_d, zero_))
        features_shadow_d = torch.from_numpy(features_shadow_d)
        features_d = features.cpu().detach()

    sample_adj,sample_features,sample_list=sample(adj_shadow,features_shadow_d.cpu().detach(),96)
    sample_adj=sample_adj.type(torch.float32)
    sample_features=sample_features.type(torch.float32)

    sample_adj_or, _, _ = sample(adj, features_d.cpu().detach(), 96)
    sample_adj_or=sample_adj_or.type(torch.float32)

    # idx_t = []
    # for i in range(idx_attack.shape[0]):
    #     if idx_attack[i] not in sample_list:
    #         idx_t.append(idx_attack[i])
    # idx_t = np.array(idx_t)
    #
    # sample_adj_test, sample_features_2 = sample_test(adj, embed.cpu().detach(), 32)
    # sample_adj_test=sample_adj_test.type(torch.float32)
    # sample_features_test=sample_features_2.type(torch.float32)








    model.attack(features, init_adj, labels, idx_attack, num_edges, epochs=args.epochs)

    inference_adj = model.modified_adj.cpu()
    # np.save('result/result_MI_%s'%args.dataset,inference_adj.numpy())
    # np.save('result/result_MI_index_%s'%args.dataset,idx_attack)
    # inference_adj = np.load('result/result_MI_{}_{}.npy'.format(args.dataset, args.defense))
    # inference_adj = torch.from_numpy(inference_adj)
    # inference_adj = np.load('result/result_MI_%s.npy' % args.dataset)
    # idx_attack = np.load('result/result_MI_index_%s.npy' % args.dataset)
    # inference_adj = torch.from_numpy(inference_adj)

    print('=== testing GCN on original(clean) graph ===')
    dtet(adj, features, labels, victim_model)
    # adj_infer_test=np.multiply(inference_adj.numpy(),for_test)
    # adj_infer_test=np.delete(adj_infer_test,list_2,axis=0)
    print('=== calculating link inference AUC&AP ===')
    metric(adj.numpy(), inference_adj.numpy(), idx_attack)




    sample_adj_test, sample_features_test = sample_infer(adj, features_d.cpu().detach(),inference_adj, 96)
    sample_adj_test=sample_adj_test.type(torch.float32)
    sample_features_test=sample_features_test.type(torch.float32)


    args_2 = parse_arguments('train_ego_small.yaml')
    ori_config_dict = get_config(args_2)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)


    sample_adj_b=train_main(config_dict,args_2,sample_adj,sample_features,sample_adj_test,sample_features_test)
    inference_adj_2=pingjie(adj,inference_adj,sample_adj_b,96)
    # metric(adj.numpy(), inference_adj_2.numpy(), idx_attack)





    # a=inference_adj.numpy()
    # a[a>0.5]=1
    # a[a<0.5]=0
    # metric(adj.numpy(), a, idx_attack)
    # metric(adj.numpy(), inference_adj.numpy(), idx_t)
    # metric_3(sample_adj_or.cpu().numpy(),sample_adj_b.cpu().numpy(), sample_list)
    sample_adj_test,sample_features_test=sample_test(adj,features.cpu().detach(),96)
    metric_3(sample_adj_test.cpu().numpy(),sample_adj_b.cpu().numpy(), sample_list)
    # metric(adj.numpy(), inference_adj.numpy(), idx_attack)




    #output = embedding(features.to(device), torch.zeros(adj.shape[0], adj.shape[0]).to(device))
    #adj1 = dot_product_decode(output.cpu())
    #metric(adj.numpy(), adj1.detach().numpy(), idx_attack)


if __name__ == '__main__':
    main()
