# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
#from decoder import VAE_RESDIUAL,VAE_ATTENTION


# %%
VAE_RESDIUAL,VAE_ATTENTION=1,1
class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_RESDIUAL(128,128),
            VAE_RESDIUAL(128,128),
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            VAE_RESDIUAL(128,256),
            VAE_RESDIUAL(256,256),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            VAE_RESDIUAL(256,512),
            VAE_RESDIUAL(512,512),
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            VAE_RESDIUAL(512,512),
            VAE_RESDIUAL(512,512),
            VAE_RESDIUAL(512,512),
            VAE_ATTENTION(512),
            VAE_RESDIUAL(512,512),
            F.group_norm(32,512),
            F.silu()
        )
    def forward(self,x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
        for module in self:
            if getattr(module,"stride",None)==(2,2):
                x=F.pad(x,[0,1,0,1])
            x=module(x)
            mean,log_varience=torch.chunk(x,2,1)
            log_varience=torch.clamp(log_varience,-30,20)
            varience=log_varience.exp()
            std=torch.sqrt(varience)
            # mean+std*sigma here sigma is noise 
            x=mean+std*noise
            x*=0.18215
            return x




