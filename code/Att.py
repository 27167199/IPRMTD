import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWeightedPooling(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(AttentionWeightedPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 2
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, M_seq):
        input_dim = len(M_seq.shape)
        if torch.isnan(M_seq).any() or torch.isinf(M_seq).any():
            if input_dim == 2:
                return torch.mean(M_seq, dim=0), None
            else:
                return torch.mean(M_seq, dim=1), None
        if input_dim == 2:
            e = self.attention_net(M_seq)
            e = e - e.max()
            alpha = F.softmax(e, dim=0)
            M = torch.sum(M_seq * alpha, dim=0)
        elif input_dim == 3:
            batch_size, num_tokens, embed_dim = M_seq.shape
            M_seq_flat = M_seq.reshape(-1, embed_dim)
            e = self.attention_net(M_seq_flat)
            e = e.view(batch_size, num_tokens, 1)
            e = e - e.max(dim=1, keepdim=True)[0]
            alpha = F.softmax(e, dim=1)
            M = torch.sum(M_seq * alpha, dim=1)
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}")
        return M, alpha
