import torch
import torch.nn as nn
from mamba_ssm import Mamba

import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

class TemporalTokenEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TemporalTokenEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        # padded_input, lengths = rnn_utils.pad_packed_sequence(x, batch_first  = True)
        # padded_inputs = padded_input.to("cuda:0")
        # return self.embedding(padded_input), lengths
        return self.embedding(x)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaBlock, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_forward = self.mamba(x)

        # Backward Mamba
        x_flip = torch.flip(x, dims=[1])  # Flip Sequence
        X_backward = self.mamba(x_flip)
        X_backward = torch.flip(X_backward, dims=[1])  # Flip back
        # print("mamba out backward shape :", mamba_out_backward.shape)
    
        # Combining forward and backward
        Y_cap = x_forward + X_backward
        Y_mlp = self.feed_forward(Y_cap)
        Y_layer_norm  = self.norm2(Y_mlp)
        # print("mamba out 1 shape :", mamba_out1.shape)
    
        X_curr  = Y_cap + Y_layer_norm

        # ff_out  = mamba_out + mamba_out2
        # output = ff_out + residualstart_index
        return X_curr



class VanillaMambaBlock(nn.Module):
    def __init__(self, d_model, n_state):
        super(VanillaMambaBlock, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        # x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x)

    
        mamba_out = self.feed_forward(mamba_out_forward)

        # ff_out  = mamba_out + mamba_out2
        # output = ff_out + residualstart_index
        return mamba_out



class BiMambaEncodingLayer(nn.Module):
    def __init__(self, embedding_dim, num_blocks):
        super(BiMambaEncodingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([BiMambaBlock(embedding_dim, embedding_dim) for _ in range(num_blocks)])

        self.mamba_block = BiMambaBlock(embedding_dim, embedding_dim)
    def forward(self, x):
        # print(" embedding dimension is : ", self.embedding_dim)
        for block in self.blocks:
            x = block(x)
        # x = self.mamba_block(x)
        # x = self.mamba_block(x)
        
            # x = block(x)
            # print(" x shape in mamba block is : ", x.shape)
        
        return x
    
    
class VanillaMambaEncodingLayer(nn.Module):
    def __init__(self, embedding_dim, num_blocks = 2):
        super(VanillaMambaEncodingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([VanillaMambaBlock(embedding_dim, embedding_dim) for _ in range(num_blocks)])

        self.mamba_block = VanillaMambaBlock(embedding_dim, embedding_dim)
    def forward(self, x):
        # print(" embedding dimension is : ", self.embedding_dim)
        for block in self.blocks:
            x = block(x)
        # x = self.mamba_block(x)
        # x = self.mamba_block(x)
        
            # x = block(x)
            # print(" x shape in mamba block is : ", x.shape)
        
        return x
    
    

class BBoxLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BBoxLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        # self.mamba = Mamba(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.feed_forward(out)
    
        # Take the output from the last time step
        out = self.fc(out[:, -1, :]).float()
        

        return out
    
    

class FullModelMambaOffset(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_blocks, prediction_dim, mamba_type= "bi mamba"):
        super(FullModelMambaOffset, self).__init__()
        self.mamba_type = mamba_type
        self.temporal_token_embedding = TemporalTokenEmbedding(input_dim, embedding_dim)
        self.bi_mamba_encoding_layer = BiMambaEncodingLayer(embedding_dim, num_blocks)
        self.vanilla_mamba_encoding_layer = VanillaMambaEncodingLayer(embedding_dim, num_blocks = 2)
        self.prediction_head = nn.Linear(embedding_dim, prediction_dim)

    def forward(self, x):
        x = self.temporal_token_embedding(x)
        # print(" x shape is : ", x.shape)
        # print(' type of x is : ', type(x))
        # x  =  x.unsqueeze(0)
        # print(" x after reshaping it is : ", x.shape)
        if self.mamba_type == "vanilla mamba":
            x = self.vanilla_mamba_encoding_layer(x)
        elif self.mamba_type == "bi mamba":    
            x = self.bi_mamba_encoding_layer(x)
        # print(" x shape after  bimamba encoding layer is : ", x.shape)
        # x = self.prediction_head(x) ## This returns (batch, contect_window, 4) where 4 is the bounind box
        
        # We only want the last element prediction
        x = self.prediction_head(x[:, -1, :]) 
        
        # print(" x shape after  prediction head layer : ", x.shape)
        
        return x
    
    

class FullModelMambaBBox(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_blocks, prediction_dim, mamba_type):
        super(FullModelMambaBBox, self).__init__()
        self.mamba_type = mamba_type
        self.temporal_token_embedding = TemporalTokenEmbedding(input_dim, embedding_dim)
        self.vanilla_mamba_encoding_layer = VanillaMambaEncodingLayer(embedding_dim, num_blocks)
        self.bi_mamba_encoding_layer = BiMambaEncodingLayer(embedding_dim, num_blocks)

        self.prediction_head = nn.Linear(embedding_dim, prediction_dim)
        self.end_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.temporal_token_embedding(x)
        # print(" x shape is : ", x.shape)
        # packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first = True)
        # print(' type of x is : ', type(x))
        # x  =  x.unsqueeze(0)
        # print(" x after reshaping it is : ", x.shape)
        if self.mamba_type == "vanilla-mamba":
            x = self.vanilla_mamba_encoding_layer(x)
        elif self.mamba_type == "bi-mamba":    
            x = self.bi_mamba_encoding_layer(x)
        
        # print(" x shape after  bimamba encoding layer is : ", x.shape)
        # x = self.prediction_head(x) ## This returns (batch, contect_window, 4) where 4 is the bounind box
        
        # We only want the last element prediction
        x = self.prediction_head(x[:, -1, :]) 
        x = self.end_activation(x)   ## Adding this to keep the output between 0 and 1 since we're calculating GiOU loss/CioU Loss
        
        # print(" x shape after  prediction head layer : ", x.shape)
        
        return x