from model import GCN
from util import accuracy
from torch_geometric.datasets import Planetoid, NELL
import torch
import torch_geometric as tg
import torch_geometric.utils as tgu
import torch.sparse as tss
import torch.optim as optim
import torch.nn.functional as F
import time



data = Planetoid(root = "./data/cora", name = 'cora')
graph = data[0]

adj_graph = tgu.to_torch_coo_tensor(graph['edge_index'])

adj_graph_loop = tgu.add_self_loops(adj_graph)

adj = torch.mm(
    torch.diag((torch.pow(tss.sum(adj_graph_loop[0], dim = 1), -1)).to_dense()).to_sparse_coo(),
    adj_graph_loop[0]
)

features = graph['x']
rowsum = features.sum(axis = 1)
rowsum_inv = torch.diag(1/rowsum)
features = torch.einsum('ij, jk -> ik', rowsum_inv, features)

labels = graph['y']
idx_train = graph['train_mask']
idx_val = graph['val_mask']
idx_test = graph['test_mask']


# Model
model = GCN(input_dim = features.shape[1],
            hidden_dim = 16,
            num_class = 7,
            dropout = 0.5)

optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

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
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()