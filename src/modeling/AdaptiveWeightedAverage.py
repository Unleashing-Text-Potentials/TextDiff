import torch
import torch.nn as nn

class AdaptiveWeightedAverage(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.weight_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, diff_feat):
        # diff_feat: [batch_size, num_diff, dim]
        weights = self.weight_net(diff_feat)  # [batch_size, num_diff, 1]
        aggregated = torch.sum(weights * diff_feat, dim=1)  # [batch_size, dim]
        return aggregated