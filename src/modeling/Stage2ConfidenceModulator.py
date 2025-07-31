import torch
import torch.nn as nn
import torch.nn.functional as F 
class Stage2ConfidenceModulator(nn.Module):
    def __init__(self, text_dim=512, diff_dim=512, hidden_dim=64, max_value=0.2):
        super().__init__()
        self.max_value = max_value
        

        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(diff_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(3)
        ])
        

        self.fusion_net = nn.Sequential(
            nn.Linear(text_dim + hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, diff_feat, stage1_feat):
        
        scale_features = []
        for net in self.scale_nets:

            transformed = net(diff_feat)

            # pooled = torch.max(transformed, dim=1)[0]
            scale_features.append( transformed )
        

        multi_scale = torch.cat(scale_features, dim=1)
        

        fused = torch.cat([stage1_feat, multi_scale], dim=1)
        

        weight = self.fusion_net(fused) * self.max_value
        return weight