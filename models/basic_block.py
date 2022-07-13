import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp


#===============================================================
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = GCNConv(inplanes, planes)
        self.conv2  = GCNConv(planes, planes)
        self.relu   = nn.ReLU(inplace=True)
    #---------------------------------------
    def forward(self, x, edge_index):
        out   = self.conv1(x, edge_index)
        out1  = self.relu(out)
        out   = self.conv2(out1, edge_index)
        out2  = self.relu(out)
        return out2


#===============================================================
class Classification_Module(nn.Module):
    def __init__(self, num_features_xd=256, output_dim=1, dropout = 0):
        super(Classification_Module, self).__init__()
        self.fc_g1   = torch.nn.Linear(num_features_xd, num_features_xd)
        self.fc_g2   = torch.nn.Linear(num_features_xd, num_features_xd//2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(num_features_xd//2, output_dim)
        self.sigmoid = nn.Sigmoid()
    #---------------------------------------
    def forward(self, x, batch):
        # Maxpool
        x   = gmp(x, batch)
        #MLP
        x   = self.relu(self.fc_g1(x))
        x   = self.dropout(x)
        x   = self.fc_g2(x)
        x   = self.dropout(x)
        out = self.out(x)
        out = self.sigmoid(out)
        return out
