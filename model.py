import dgl.nn.pytorch as dglnn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn import GATConv, SAGEConv, GINConv





class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes,dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        self.layers.append(
            dglnn.GraphConv(hidden_dim//2, hidden_dim//2,  activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hidden_dim//2, n_classes))
        
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, g, features):
        h=self.feature_extractor(features)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        logits=None
        with g.local_scope():
            g.ndata['h'] = h
            logits = dgl.readout_nodes(g, 'h', op='mean')

        return logits


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes,dropout_rate=0.5, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        for i in range(n_layers-1):
            self.layers.append(
                dglnn.GINConv(nn.Linear(hidden_dim//2, hidden_dim//2))
            )
        self.layers.append(dglnn.GINConv(nn.Linear(hidden_dim//2, n_classes)))
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.n_layers = n_layers

    def forward(self, g, features):
        h=self.feature_extractor(features)
        for i in range(self.n_layers - 1):
            h = F.relu(self.layers[i](g, h))
        h = self.layers[-1](g, h) 
        logits=None
        with g.local_scope():
            g.ndata['h'] = h
            logits = dgl.readout_nodes(g, 'h', op='mean')

        return logits



class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout_rate=0.5, num_heads=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )


        self.layers.append(dglnn.GATConv(hidden_dim//2,  hidden_dim//2, num_heads=num_heads, activation=F.relu))
        self.layers.append(dglnn.GATConv(hidden_dim//2*num_heads, n_classes, num_heads=1, activation=None))
        

    def forward(self, g, features):
        h=self.feature_extractor(features)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != (len(self.layers)-1):
                h = h.flatten(1) # (N, num_heads * out_dim)
            else:
                h = h.squeeze(1) # (N, n_classes)
        logits=None
        with g.local_scope():
            g.ndata['h'] = h
            logits = dgl.readout_nodes(g, 'h', op='mean')
        return logits





class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout_rate=0.25):
        super(MLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, n_classes)
        )


    def forward(self, g, h):
        h = self.feature_extractor(h) 
        
        h=self.mlp(h)
        logits=None
        with g.local_scope():
            g.ndata['h'] = h
            logits = dgl.readout_nodes(g, 'h', op='mean')
        return logits




class SAGE(nn.Module):#no dropout
    def __init__(self, input_dim, hidden_dim, n_classes, dropout_rate=0.5, n_layers=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(hidden_dim//2, hidden_dim//2, aggregator_type='gcn'))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden_dim//2, hidden_dim//2, aggregator_type='gcn'))
            self.convs.append(SAGEConv(hidden_dim//2, n_classes, aggregator_type='gcn'))

    def forward(self, g, h):
        h = self.feature_extractor(h) 
        for i in range(self.n_layers - 1):
            h = F.relu(self.convs[i](g, h))
        h = self.convs[-1](g, h) 
        logits=None
        with g.local_scope():
            g.ndata['h'] = h
            logits = dgl.readout_nodes(g, 'h', op='mean')
        return logits




