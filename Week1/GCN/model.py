import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tss

class GraphConv(nn.Module) :

    def __init__(self, in_, out_) :
        super(GraphConv, self).__init__()
        
        self.in_features = in_
        self.out_features = out_
        
        self.weight = nn.Parameter(torch.FloatTensor(in_, out_))
        self.bias = nn.Parameter(torch.FloatTensor(out_))
        
        # Initial Weights
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, input, adj_graph) :
        
        output = tss.mm(input, self.weight)
        output = tss.mm(adj_graph, output) + self.bias
        
        return output 
        
class GCN(nn.Module) :
    
    def __init__(self,
                 input_dim, 
                 hidden_dim, 
                 num_class,
                 dropout = 0.5) :
        super(GCN, self).__init__()
        
        self.GC1 = GraphConv(input_dim, hidden_dim)
        self.GC2 = GraphConv(hidden_dim, num_class)
        self.dropout = dropout
        
    def forward(self, input, adj_graph) :
        output = F.relu(self.GC1(input, adj_graph))
        output = F.dropout(output, self.dropout, training = self.training)
        output = self.GC2(output, adj_graph)
        
        return F.log_softmax(output, dim = 1)
    