import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

## Example data: Nodes with attributes

data = {
    'node_id':[0,1,2,3],
    'age': [25, 30, 22, 28],
    'location' : ['New York', 'San Fransisco', 'Log Angeles', 'New York'],
    'profession' : ['Engineer', 'Designer', 'Artist', 'Engineer']
}

df = pd.DataFrame(data)

## Normalize continuous features
scaler = StandardScaler()
df['age'] = scaler.fit_transform(df[['age']])
print("Normalized age : ", df['age'])


## One-hot encode categorical features (location, profession)


encoder = OneHotEncoder(sparse_output= False)
encoded_location = encoder.fit_transform(df[['location']])
encoded_profession = encoder.fit_transform(df[['profession']])
print("One hot encoded location : ", encoded_location)
print("One hot encoded profession : ", encoded_profession)


# age = df[['age']].astype(float)
# encoded_location = pd.DataFrame(encoded_location).astype(float)
# encoded_profession = pd.DataFrame(encoded_profession).astype(float)

##Combine all features into a single tensor
# node_features = torch.tensor(pd.concat([df[['age']], pd.DataFrame(encoded_location), pd.DataFrame(encoded_profession)], axis = 1).values, torch.float)
node_features = torch.tensor(pd.concat([df[['age']], pd.DataFrame(encoded_location), pd.DataFrame(encoded_profession)], axis=1).values, dtype=torch.float)


print(" Node features : {}".format(node_features))


### Create Custom Dataset
## Now that you have node features, you need to create a graph dataset where these features are associated with each node

from torch_geometric.data import Data

##Example edge list : 0 <--> 1, 1<--> 0, 2 <--> 3
# 1 and 0 are connected together, 2 and 3 are connected together
edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 0, 3, 2]], dtype = torch.long)


## Create a pytorch geometric data object

graph_data = Data(x = node_features, edge_index = edge_index)

print(graph_data)

## Define a GNN model using GNN node Features
## We'll define a simple GNN model that takes the custom node features as input


from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x  = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index) ### HERE WE PASS EDGE INDEX TO LET THE MODEL KNOW ABOUT THE NEIGHBOURING NODES FOR MESSAGE PASSING
        return x

### Define the model
model = GNN(in_channels = node_features.size(1), hidden_channels = 16, out_channels = 2) ## Here we're assuming 2 output classes.

## print the model
print(model)




## Train the model

## Finally, we'll train the GNN model our custom node features

## Dummy labels (for illustraion purposes)
labels  = torch.tensor([0,1,0,1], dtype = torch.long)

## Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

## Training loop
model.train()

for epoch in range(100):
    optimizer.zero_grad()

    #Forward pass
    out = model(graph_data.x, graph_data.edge_index)
    # print("node embeddings is : ", out)
    # print("labels are : ", labels)
    ##Compute loss (assume a simple node classification task)
    loss = criterion(out, labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 ==0:
        print("Epoch : {}, Loss : {}".format(epoch-1, loss.item()))




