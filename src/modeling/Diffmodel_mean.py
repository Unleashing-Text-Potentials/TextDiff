import torch
import torch.nn as nn

class LatentProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 4x64x64
            nn.Conv2d(4, 128, kernel_size=3, stride=2, padding=1),  #  128x32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #  256x16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), #  512x8x8
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  #  512x1x1

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 512
    

class StochasticText_mean(nn.Module):
    def __init__(self):
        super(StochasticText_mean, self).__init__()

        self.embed_dim = 512

        # self.diffision = run_diffusion()

        self.diff_linear = LatentProjector()


    def forward(self, images = None , text_features = None , video_features = None):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_texts x embed_dim
        """
        # print( text_features.dtype )
        # print( images.dtype )
        # ls
        # print( images.shape )
        batch = text_features.shape[0]
        n = images.shape[1]
        

        # images = images.flatten(start_dim=2)  # [batch, 4 * 64 * 64=16384]
        _ , _ , a , b , c = images.shape
        
        images = images.view(batch*n , a , b , c )
        log_var = self.diff_linear( images )
        log_var = log_var.view( batch , n , -1 )

        
        return log_var


