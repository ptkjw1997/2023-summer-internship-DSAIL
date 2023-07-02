from model import GCN
from util import accuracy, EarlyStop
from torch_geometric.datasets import Planetoid, NELL
import torch
import torch_geometric as tg
import torch_geometric.utils as tgu
import torch.sparse as tss
import torch_sparse
import torch.optim as optim
import torch.nn.functional as F
import time



#data = Planetoid(root = "./data/Cora", name = 'Cora')
#data = Planetoid(root = "./data/CiteSeer", name = 'CiteSeer')
#data = Planetoid(root = "./data/PubMed", name = 'PubMed')
data = NELL(root = "./data/NELL")
graph = data[0]


# Normalize Graph
adj_graph = tgu.to_torch_coo_tensor(graph['edge_index'])

adj_graph_loop = tgu.add_self_loops(adj_graph)[0]
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
#idx_train = graph['train_mask']
#idx_val = graph['val_mask']
#idx_test = graph['test_mask']

idx_train = range(100)
idx_val = range(200, 500)
idx_test = range(500, 1500)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
    
# Model
model = GCN(input_dim = features.shape[1],
            hidden_dim = 64,
            num_class = 186,
            dropout = 0.1)

optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=1e-5)

early_stop = EarlyStop()

if torch.cuda.is_available() :
    use_cuda = True
else :
    use_cuda = False
    
if use_cuda :
    print("Use Cuda")
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
def train(epoch):
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
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    early_stop(loss_val)
    if early_stop.is_stop :
        return True
    else :
        return False

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(200):
    need_stop = train(epoch)
    if early_stop.is_best :
        print("Early Test")
        test()
    if need_stop :
        break
    
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()