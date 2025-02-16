import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

class SelectiveSSM(nn.Module):
    """Selective State Space sequence model component"""
    def __init__(self, d_model, d_state=16, dropout=0.0):
        super().__init__()
        
        # SSM parameters
        self.d_model = d_model
        self.d_state = d_state
        
        # Linear projections for A, B, C, D matrices
        self.A = nn.Linear(d_model, d_state, bias=False)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Selective gating
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch, self.d_state, device=x.device)
        
        # Output container
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            # Current input
            xt = x[:, t, :]
            
            # Compute gate
            g = torch.sigmoid(self.gate(xt))
            
            # Update state
            h = torch.tanh(self.A(xt)) * h + self.B(xt)
            
            # Compute output
            yt = self.C(h) * g + self.D(xt) * (1 - g)
            
            outputs.append(yt)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        
        return self.dropout(output)

class MambaBlock(nn.Module):
    def __init__(self, hidden_size, d_state, dropout_rate):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        self.ssm = SelectiveSSM(hidden_size, d_state, dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        # SSM branch
        y = self.norm(x)
        y = self.ssm(y)
        x = x + y
        
        # FFN branch
        y = self.norm(x)
        y = self.ffn(y)
        x = x + y
        
        return x

class MambaModel(nn.Module):
    def __init__(
        self,
        hops,
        n_class,
        input_dim,
        n_layers=6,
        hidden_dim=64,
        d_state=16,
        dropout_rate=0.0
    ):
        super().__init__()
        
        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, n_class)
        
        # Node attention
        self.node_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize parameters
        self.apply(init_params)
        
    def forward(self, batched_data):
        # Initial projection
        x = self.input_proj(batched_data)
        
        # Mamba layers
        for layer in self.layers:
            x = layer(x)
            
        # Final normalization
        x = self.final_ln(x)
        
        # Split into node and neighbors
        node = x[:, 0:1, :]
        neighbors = x[:, 1:, :]
        
        # Compute attention weights
        node_expanded = node.expand(-1, self.seq_len-1, -1)
        attn_input = torch.cat((node_expanded, neighbors), dim=2)
        attention_weights = self.node_attention(attn_input)
        
        # Apply attention
        neighbor_aggregated = torch.sum(neighbors * attention_weights, dim=1, keepdim=True)
        
        # Combine node and aggregated neighbors
        output = (node + neighbor_aggregated).squeeze(1)
        
        # Final classification
        output = self.classifier(torch.relu(self.out_proj(output)))
        
        return torch.log_softmax(output, dim=1)