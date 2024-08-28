## Define node and edge features
## Lets start by defining a somple graph with node and edge features

import torch
from torch_geomtric.data import Data

## Node features : 4 nodes with 3 dimensional features
node_features  = torch.tensor([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype = torch.float)


## Edge indices : Defined connection between nodes

edge_index = torch.tensor([[0, 1, 2, 3],
                           1, 2, 0, 0], dtype = torch.long])


## Edge features: 4 edges with 2 dimensional features
edge_features = torch.tensor([[0.5, 1.0], [0.3, 0.7], [0.9, 0.1], 0.4, 0.8])

## Create a graph data object
data = Data(x = node_features, edge_index = edge_index, edge_attr = edge_features)

## Step 2: Create a GN model with edge features
## Next we'll create a simple GNN model that incorporates edge features. PyTorch
## Gewometric's 'MessagePassing' class allows us to easily incorporate edge features in the 
## Message passing process.



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GNNWithEdgeFeatures(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        super(GNNWithEdgeFeatures, self).__init__(aggr = "mean") ## Mean aggregation
        self.node_lin = nn.Linear(node_in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_in_channels, out_channels)
        self.final_lin = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        ## x : node features [num_nodes, node_in_channales]
        ## Edge index : Graph connectivity [2, num_edges]
        ## edge_attr : Edge features [num_edges, edge_in_channels]

        ## Initial transformation of node and edfe features
        x = self.node_lin(x)
        edge_attr = self.edge_lin(edge_attr)

        #Propagation information through the graph
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)
    

    def message(self, x_j, edge_attr):
        ## Message function to combine the node and edge features
        return se