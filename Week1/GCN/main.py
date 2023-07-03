from model import GCN
from util import accuracy, EarlyStop, load_data
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

def do(data_name) :
    # Basic Variable
    data_name = data_name
    data_balance = True
    use_early_stop = False

    torch.manual_seed(67) 

    # Hyper Parameter Setting
    if data_name != "NELL" :
        hidden_dim = 16
        lr = 0.01
        dropout = 0.5
        weight_decay = 5e-4
    else :
        hidden_dim = 64
        lr = 0.01
        dropout = 0.1
        weight_decay = 1e-5
        
    adj, features, labels, idx_train, idx_val, idx_test = load_data(data_name,
                                                                    data_balance)
    n_class = labels.max().item() + 1
    
    # Model
    model = GCN(input_dim = features.shape[1],
                hidden_dim = hidden_dim,
                num_class = n_class,
                dropout = dropout)

    optimizer = optim.Adam(model.parameters(),
                        lr=lr, weight_decay=weight_decay)

    early_stop = EarlyStop()

    if torch.cuda.is_available() :
        use_cuda = True
    else :
        use_cuda = False
        
    if use_cuda :
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    def train(epoch, verbose = False):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if verbose :
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))
        
        
        early_stop(loss_val)
        if use_early_stop and early_stop.is_stop :
            return True
        else :
            return False

    def test(verbose = True):
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        
        if verbose :
            print("Test Set Result : After Train ",
                f"loss= {loss_test.item():.4f}",
                f"accuracy= {acc_test.item():.4f}\n",
                sep = "\n")
        
        return loss_test, acc_test

    # Train model
    t_total = time.time()
    for epoch in range(200):
        need_stop = train(epoch)
        if early_stop.is_best :
            model.eval()
            loss_test, acc_test = test(verbose = False)
            early_stop.best_model(loss_test, acc_test)
        if need_stop :
            break
    
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(f"Train Finished : {data_name}")
    test()
    print(f"Best Model Loss : {early_stop.best_loss_test:.4f}",
        f"Best Model Acc : {early_stop.best_acc_test:.4f}\n",
        sep = "\n")


if __name__ == "__main__" : 
    for name in ["Cora", "Citeseer", "PubMed", "NELL"] :
        do(name)