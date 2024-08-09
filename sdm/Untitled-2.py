# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
#from attention import SelfAttention


# %%
class VAE_Residual(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.gnorm1=nn.GroupNorm(32,in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.gnorm2=nn.GroupNorm(32,out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if in_channels==out_channels:
            self.residual=nn.Identity()
        else:
            self.residual=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residue=x
        x=self.gnorm1(x)
        x=self.conv1(x)
        x=F.silu(x)
        x=self.gnorm2(x)
        x=self.conv2(x)
        x=F.silu(x)
        return x+self.residual(residue)

# %%
SelfAttention=1
class VAE_Attention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.gnorm=nn.GroupNorm(32,channels)
        self.attetion=SelfAttention(1,channels)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        resdiue=x
        b,c,w,h=x.shape
        x=x.view(b,c,w*h)
        x=x.transpose(-1,-2)
        x=self.attetion(x)
        x=x.transpose(-1,-2)
        x=x.view(b,c,w,h)
        x=x+resdiue
        return x

# %%
class Decoder(nn.ModuleList):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_Residual(512,512),
            VAE_Attention(512,512),
            VAE_Residual(512,512),
            VAE_Residual(512,512),
            VAE_Residual(512,512),
            VAE_Residual(512,512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_Residual(512,512),
            VAE_Residual(512,512),
            VAE_Residual(512,512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_Residual(512,256),
            VAE_Residual(512,256),
            VAE_Residual(512,256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_Residual(256,128),
            VAE_Residual(256,128),
            VAE_Residual(256,128),
            nn.GroupNorm(32,128),
            F.silu(),
            nn.Conv2d(128,3,kernel_size=3,padding=1)
          
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x is the shape of batch_size,512,channels h/8,w/8
        x/=0.18215
        for module in self:
            x=module(x)
        return x



