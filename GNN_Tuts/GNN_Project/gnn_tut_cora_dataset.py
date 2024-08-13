import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader


## Load the graph dataset
## We'll use the Coda dataset, which is a citation network where nodes represent documents and edges represent citations


dataset = Planetoid(root = "/tmp/Coda", name = "Cora")

## Inspect the dataset

print("Dataset : {}".format(dataset))
print("Number of graphs : {}".format(len(dataset)))
print("Number of features : {}".format(dataset.num_features))
print(" Number of classes : {}".format(dataset.num_classes))


## Use the first graph in the dataset for your work
data = dataset[0]


## print data details now

print(data)


### Define the GraphSAGE model
## GraphSAGE aggregates node information from its neighbours to generate node embeddings



class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, 128)

        self.conv3 = SAGEConv(128, out_channels) ### This is the the final layer        

    def forward(self, x, edge_index):

        ## First ;ayer: apply SAGEConv and ReLU Activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        ### Second layer: apply SAGEConv
        x  = self.conv2(x, edge_index)   ## this layer is basically the output layer only which returns the embedding for classification task
        x = F.relu(x)

        x  = self.conv3(x, edge_index)   
        # x = F.relu(x)


        return x
    

### Initialize and train the model
## We'll train the model to generate node embeddings that can be used for node classification



##Defin the model, loss function, and optimizer
model = GraphSAGE(in_channels = dataset.num_features, hidden_channels = 64, out_channels = dataset.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

## Training loop
model.train()
for epoch in range(200): ## Train for 200 epochs
    optimizer.zero_grad()

    ## Forward pass
    out = model(data.x, data.edge_index)
    

    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    ## Backward pass and optimization
    loss.backward()
    optimizer.step()

    if epoch %10 ==0:
        print("Epoch : {}, Loss : {}".format(epoch+1, loss.item()))




## Extract the node embeddings:
## Once the model is trained, we can extract the node embeddings from the mode

## Switch the evaluation mode
model.eval()


## Extract embeddings for all nodes
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)


## Print the shape of embeddings
print(" Node embedding shape is : ",embeddings.shape)

## Visualize of use node embeddings

## You can use the extracted node embeddings for various downstream tastks.
## For example: you can visualize them using t-SNE os use them for node classification


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embeddings_2d = TSNE(n_components = 2).fit_transform(embeddings.cpu().numpy())

## Plot the 2D embeddings, color-coded by class labels
plt.figure(figsize = (8, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c = data.y.cpu(), cmap = 'Spectral', s = 10)
plt.colorbar()
plt.title("Node EMbedding Visualization")
# plt.show()
plt.savefig("node_embedding_cora_visualization.png")