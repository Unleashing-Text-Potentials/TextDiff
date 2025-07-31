import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = 512
        self.num_heads = 1
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class ChannelGatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  
        )

        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, attn_feat, text_feat):

        combined = torch.cat([attn_feat, text_feat], dim=-1)  # (batch, n, 2*embed_dim)
        
        gate = self.gate_proj(combined)  # (batch, n, embed_dim)

        fused = gate * attn_feat + (1 - gate) * text_feat

        output = fused + self.scale * text_feat
        
        return output

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embed_dim = 512
        dropout = 0.3

        self.cross_attn = MultiHeadedAttention()

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.channel = ChannelGatedFusion(self.embed_dim)
        self.dropout_diff = nn.Dropout(p=0.1)
        self.layer_norm_diff = nn.LayerNorm(self.embed_dim)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # print( out.shape)

        batch_size = text_embeds.size(0)
        diag_idx = torch.arange( batch_size, device=text_embeds.device)
        diag_attn = out[diag_idx, diag_idx, :]

        fused_feat = self.channel(diag_attn, text_embeds)
        # print( fused_feat.shape)
        fused_feat = self.layer_norm_diff(fused_feat)

        return fused_feat
