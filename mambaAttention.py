import torch
import torch.nn as nn
from mamba_ssm import Mamba

class TemporalTokenEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TemporalTokenEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class MambaSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_state, num_heads):
        super(MambaSelfAttentionBlock, self).__init__()
        self.mamba = Mamba(d_model, n_state)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Mamba processing
        mamba_out = self.mamba(x)
        mamba_out = self.norm1(mamba_out + x)
        
        # Self-attention processing
        attn_out, _ = self.self_attention(mamba_out, mamba_out, mamba_out)
        attn_out = self.norm2(attn_out + mamba_out)
        
        # Feed-forward network
        ff_out = self.feed_forward(attn_out)
        out = self.norm3(ff_out + attn_out)
        
        return out

class CombinedEncodingLayer(nn.Module):
    def __init__(self, embedding_dim, num_blocks, num_heads):
        super(CombinedEncodingLayer, self).__init__()
        self.blocks = nn.ModuleList([MambaSelfAttentionBlock(embedding_dim, embedding_dim, num_heads) for _ in range(num_blocks)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class FullModelMambaBBoxAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_blocks, prediction_dim, num_heads, mamba_type = "bi-mamba"):
        super(FullModelMambaBBoxAttention, self).__init__()
        self.temporal_token_embedding = TemporalTokenEmbedding(input_dim, embedding_dim)
        self.combined_encoding_layer = CombinedEncodingLayer(embedding_dim, num_blocks, num_heads)
        self.prediction_head = nn.Linear(embedding_dim, prediction_dim)
        self.end_activation = nn.Sigmoid()
        # print("Loaded model attention mamba")

    def forward(self, x):
        x = self.temporal_token_embedding(x)
        x = self.combined_encoding_layer(x)
        
        # We only want the last element prediction
        x = self.prediction_head(x[:, -1, :])
        x = self.end_activation(x)
        
        return x