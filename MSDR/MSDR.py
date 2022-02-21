import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MSDRCell(nn.Module):
    def __init__(self, num_nodes, hidden_size, pre_k, pre_v):
        super(MSDRCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.pre_v = pre_v
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_nodes, hidden_size), requires_grad=True)
        self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, hidden_size), requires_grad=True)
        self.attlinear = nn.Linear(num_nodes * hidden_size, 1)

    def forward(self, inputs: Tensor, supports: Tensor, hidden_states):
        bs, k, n, d = hidden_states.size()
        _, _, f = inputs.size()
        preH = hidden_states[:, -1:]
        for i in range(1, self.pre_v):
            preH = torch.cat([preH, hidden_states[:, -(i + 1):-i]], -1)
        preH = preH.reshape(bs, n, d * self.pre_v)
        new_states = hidden_states + self.R.unsqueeze(0)
        output = torch.matmul(inputs, self.W) + self.b.unsqueeze(0) + self.attention(new_states)
        output = output.reshape(bs, 1, n, d)
        x = hidden_states[:, 1:k]
        hidden_states = torch.cat([x, output], dim=1)
        output = output.reshape(bs, n, d)
        return output, hidden_states

    def attention(self, inputs: Tensor):
        bs, k, n, d = inputs.size()
        x = inputs.reshape(bs, k, -1)
        out = self.attlinear(x)
        weight = F.softmax(out, dim=1)
        outputs = (x * weight).sum(dim=1).reshape(bs, n, d)
        return outputs
