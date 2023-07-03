from model import GCN
from torch_geometric.datasets import Planetoid, NELL
import torch
import torch_geometric as tg
import torch_geometric.utils as tgu
import torch.sparse as tss
import torch_sparse
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np

class EarlyStop(object) :
    
    def __init__(self,
                 patience = 10) :
        self.patience = patience
        self.is_stop = False
        self.counter = 0
        self.best_loss = 1e5
        self.is_best = False
        
        
    def __call__(self, loss) :
        if loss < self.best_loss :
            self.best_loss = loss
            self.counter = 0
            self.is_best = True
        else :
            self.counter += 1
            self.is_best = False
            
        if self.counter >= self.patience :
            self.is_stop = True
            
    def best_model(self, loss_test, acc_test) :
        self.best_loss_test = loss_test
        self.best_acc_test = acc_test
            
def show_data_info(idx, labels, name) :
    n_class = labels.max().item() + 1
    counter = [0] * n_class
    for i in idx :
        counter[labels[i].item()] += 1
    
    print(f"{name} Label Count : {counter}\n")
            
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(data_name, data_balance) :
    if data_name != "NELL" :
        data = Planetoid(root = f"./data/{data_name}", name = f'{data_name}')
    else :
        data = NELL(root = "./data/NELL")
    
    graph = data[0]


    # Normalize Graph
    adj_graph = tgu.to_torch_coo_tensor(graph['edge_index'])

    adj_graph_loop = tgu.add_self_loops(adj_graph)[0] # Add Self-Loop
    D = tss.sum(adj_graph_loop, dim = 1)
    indices_diag = torch.stack([D.indices(), D.indices()]).reshape(2, -1)
    D_inv = torch.sparse_coo_tensor(indices = indices_diag,
                                            values = 1.0 / D.values())
    adj = tss.mm(D_inv, adj_graph_loop)

    # Noramlize Features
    if type(data) == NELL :
        features = graph['x'].to_torch_sparse_coo_tensor() 
        rowsum = tss.sum(features, dim = 1)
        indices_diag = torch.stack([rowsum.indices(), rowsum.indices()]).reshape(2, -1)
        
        rowsum_inv = torch.sparse_coo_tensor(indices = indices_diag,
                                            values = 1.0 / rowsum.values())
        features = tss.mm(rowsum_inv, features)
    else :
        features = graph['x'].to_sparse_coo()
        rowsum = tss.sum(features, dim = 1)
        indices_diag = torch.stack([rowsum.indices(), rowsum.indices()]).reshape(2, -1)

        rowsum_inv = torch.sparse_coo_tensor(indices = indices_diag,
                                            values = 1.0 / rowsum.values())
        features = tss.mm(rowsum_inv, features)
        
    labels = graph['y']
    # NELL Class Correction 
    # if data_name == "NELL" :
    #     labels_ = []
    #     labels_dict = {}
        
    #     i = 0
    #     for label in labels :
    #         if label.item() not in labels_dict :
    #             labels_dict[label.item()] = i
    #             i += 1
    #         labels_.append(labels_dict[label.item()])
    #     labels = torch.LongTensor(labels_)
        
    n_class = labels.max().item() + 1
    
    
    # Information About Data
    print(f"Data : {data_name}",
          f"Num Nodes : {adj.shape[0]}",
          f"Num Edges : {graph.num_edges}",
          f"Feature Dim : {graph.num_features}",
          f"Num Class : {n_class}\n", sep = "\n")

    if not data_balance :
        if type(data) == NELL :
            idx_train = range(100)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)
        else :
            idx_train = range(20 * n_class)
            idx_val = range(200, 500)
            idx_test = range(500, 1500)
    else :
        # Balance Train Set
        if type(data) == NELL :
            idx_train = []
            for i in range(n_class) :
                indices = torch.nonzero(labels == i).squeeze(1)
                idx_train += indices[:1].tolist()
            last_idx = max(idx_train)
            idx_val = range(last_idx+1, last_idx+301)
            idx_test = range(last_idx+501, last_idx+1501) 
        else :       
            idx_train = []
            for i in range(n_class) :
                indices = torch.nonzero(labels == i).squeeze(1)
                idx_train += indices[:20].tolist()
            last_idx = max(idx_train)
            idx_val = range(last_idx+1, last_idx+301)
            idx_test = range(last_idx+501, last_idx+1501)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    if data_name != "NELL" :
        for idx, name in zip([idx_train, idx_val, idx_test],
                             ["train", 'val', "test"]):
            show_data_info(idx, labels, name)
    
    return adj, features, labels, idx_train, idx_val, idx_test