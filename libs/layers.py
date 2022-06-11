import torch.nn as nn
import torch
from torch.nn.modules import RNN
import math

class ConstantPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(ConstantPositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.shape[1],:]

class LinearRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                ):

        super().__init__()
        
        self.input_ff = nn.Linear(input_dim, hid_dim, bias=False)
        self.hidden_ff = nn.Linear(hid_dim,hid_dim, bias=False)
        self.output_ff = nn.Linear(hid_dim, output_dim, bias=False)

        self.hid_dim = hid_dim

    def forward(self, x):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]

        hidden = []
        hidden.append(torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device))
        
        x = self.input_ff(x)

        for i in range(length):
            h_next = x[:,i:i+1,:] + self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)
        out = self.output_ff(hidden)
        return out
    

class LinearRNNEncDec(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 out_len
                ):

        super().__init__()
        
        self.U = nn.Linear(input_dim, hid_dim, bias=False)
        self.W = nn.Linear(hid_dim,hid_dim, bias=False)
        self.V = nn.Linear(hid_dim,hid_dim, bias=False)
        self.M = nn.Linear(hid_dim,hid_dim, bias=False)
        self.cT = nn.Linear(hid_dim, output_dim, bias=False)
        self.out_len = out_len
        self.hid_dim = hid_dim

    def forward(self, x):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]

        h_next=torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device)
        
        x = self.U(x)

        for i in range(length):
            h_next = x[:,i:i+1,:] + self.W(h_next)
        
        g = self.M(h_next)

        output = [g]
        for i in range(self.out_len):
            g = self.V(output[i])
            output.append(g)

        output = torch.cat(output[1:], dim=1)
        out = self.cT(output)
        return out