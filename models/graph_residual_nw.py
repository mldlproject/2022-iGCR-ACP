import os
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import torch
from models.basic_block import *


#===============================================================
class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers, filt_sizes):
        self.inplanes = 64
        super(Deeplab_ResNet_Backbone, self).__init__()
        # seed
        self.conv1 = GCNConv(39, 64)
        self.relu = nn.ReLU(inplace=True)
        #---------------------------------------
        self.filt_sizes = filt_sizes
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks) in enumerate(zip(self.filt_sizes, layers)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        #---------------------------------------
        self.blocks = nn.ModuleList(self.blocks) # contain basic block 
        self.ds = nn.ModuleList(self.ds) 
        self.layer_config = layers
    #---------------------------------------
    def _make_layer(self, block, planes, blocks): # planse: filter_size
        downsample = None
        if self.inplanes != planes:
            downsample = GCNConv(self.inplanes, planes)
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers, downsample 
    #---------------------------------------
    def forward(self, data):
        _, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.seed(data)
        for segment, num_blocks in enumerate(self.layer_config): # layer_config [2, 2, 2, 1]
            for b in range(num_blocks):
                # apply the residual skip out of _make_layers_
                if b == 0 and self.ds[segment] is not None:
                    residual = self.ds[segment](x, edge_index)
                else:
                    residual = x  
                x = F.relu(residual + self.blocks[segment][b](x, data.edge_index))
        return x
    #---------------------------------------
    def seed(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        return x


#===============================================================
class MTL2(nn.Module):
    def __init__(self, layers, num_classes_tasks, filt_sizes = [64, 128, 256, 256]):
        super(MTL2, self).__init__()
        seed = 1029
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # share networks 
        self.backbone = Deeplab_ResNet_Backbone(BasicBlock, layers, filt_sizes)
        #---------------------------------------
        # task networks
        self.num_tasks = num_classes_tasks
        for t_id in range(self.num_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(num_features_xd=filt_sizes[-1], output_dim=1))    
        self.layers = layers
    #---------------------------------------    
    def forward(self, data):
        # share networks
        feats = [self.backbone(data)] * self.num_tasks
        #---------------------------------------
        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id],data.batch)       
            outputs.append(output)
        #---------------------------------------
        return outputs


