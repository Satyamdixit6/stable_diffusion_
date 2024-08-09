# %%
import torch
import torch.nn as nn
from torch.nn import functional as F 
#import attention SelfAttention,CrossAttention

# %%
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320,4)
    def forward(self,latent,context,time):
        time=self.time_embedding(time)
        otuput=self.unet(latent,context,time)

        otuput=self.final(otuput)

        return otuput   

          







# %%
class TimeEmbedding(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.linear_1=nn.Linear(n_embd,4*n_embd)
        self.linear_2=nn.Linear(4*n_embd,4*n_embd)
    def forward(self,x):
        # x(1,320)
        x=self.linear_1(x)
        x=F.silu(x)
        x=self.linear_2(x)
        return x

# %%
class SwitchSeqential(nn.Sequential):
    def forward(self,x,context,time):
        for layer in self:
            if isinstance(layer,UNET_AttentionBlock):
                x=layer(x,context)
            elif isinstance(layer,UNET_ResidualBlock):
                x=layer(x,time)
            else:
                x=layer(x)
            return x

# %%
class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_heads:int,n_embed:int,d_context=768):
        super().__init__()
        channels=n_heads*n_embed

        self.groupnorm=nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input=nn.Conv2d(channels,channels,kernel_size=1,padding=0)


        self.layernorm_1=nn.LayerNorm(channels)
        self.attention_1=SelfAttentin(n_heads,channels,in_proj_bias=False)
        self.layernorm_2=nn.LayerNorm(channels)
        self.attention_2=CrossAttention(n_heads,channels,d_context,in_proj_bias=False)
        self.layernorm_3=nn.LayerNorm(channels)
        self.linear_geglu_1=nn.Linear(channels,4*channels*2)

        self.linear_geglu_2=nn.Linear(4*channels,channels)

        self.conv_output=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        # x:batch,features,height,width
        # y:context batch_size,seq_len,dim

        resdiue_long=x

        x=self.groupnorm(x)

        x=self.conv_input(x)

        b,c,h,w=x.shape

        x=x.view(b,c,h*w)

        x=x.transpose(-1,-2)

        resdiue_short=x
        
        x=self.layernorm_1(x)
        x=self.attention_1(x)

        x+=resdiue_short

        resdiue_short=x

        #normalizaion cross attention with context and residue

        x=self.layernorm_2=(x)
        x=self.attention_2(x,context)

        x+=resdiue_short

        resdiue_short=x

        # normalization then ffn with giglu and skip connection

        x=self.layernorm_3(x)
        # shape will batch_size,pixel=height*width,features turns in 8 times the features

        x,gate=self.linear_geglu_1(x).chunk(2,dim=-1)
        # two tensor shape of batch_size,pixcels,features/embedding
        x=x*F.gelu(gate)
        # this layers low the no of features
        x=self.linear_geglu_2(x)

        # batch_size,height*width,features
        x+=resdiue_short
        x=x.transpose(-1,-2)
        x=x.view(b,c,h,w)

        return self.conv_output(x)+resdiue_long









# %%
class UNET_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,n_time=1280):
        super().__init__()
        self.groupnorm_feature=nn.GroupNorm(32,in_channels)
        self.conv_feature=nn.Conv2d(out_channels,kernel_size=3,padding=1)
        self.linear_time=nn.Linear(n_time,out_channels)

        self.groupnorm_merged=nn.GroupNorm(32,out_channels)
        self.conv_merged=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels==out_channels:
            self.residual=nn.Identity
        else:
            self.residual=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self,features,time):
            # here the features is latent space and time is what time step it is 
        residue=features

        features=self.groupnorm(features)
        features=F.silu(features)

            # convert to same channes if in and out not match increase the no of channels
        features=self.conv_feature(features)

        time=F.silu(time)
        time=self.linear_time(time)
        # time dimmesin does not include batch_size,channles dimenssion
        merged=features+time.unsqueeze(-1).unsqueeze(-1)

        merged=self.groupnorm_merged(merged)
        merged=F.silu(merged)
        merged=self.conv_merged(merged)

        return merged+self.residual(residue)
    

# %%
class UNET(nn.Module):
    def __init__(self):
        super().__init_()
        self.encoder=nn.ModuleList([
            SwitchSeqential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSeqential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSeqential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSeqential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSeqential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,40)),
            SwitchSeqential(UNET_ResidualBlock(640,640),UNET_AttentionBlock(8,40)),
            SwitchSeqential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSeqential(UNET_ResidualBlock(640,1280),UNET_AttentionBlock(8,40)),
            SwitchSeqential(UNET_ResidualBlock(1280,1280),UNET_AttentionBlock(8,40)),
            SwitchSeqential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSeqential(UNET_ResidualBlock(1280,1280)),
            SwitchSeqential(UNET_ResidualBlock(1280,1280)),
            # 
            # bottel_neck layer




            ])
        self.bottleneck=SwitchSeqential(UNET_ResidualBlock(1280,1280),
                                        UNET_AttentionBlock(8,160),
                                        UNET_ResidualBlock(1280,1280))
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSeqential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSeqential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSeqential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSeqential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSeqential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSeqential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSeqential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSeqential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSeqential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSeqential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSeqential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSeqential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])
    def forward(self,x,context,time):
        # x=batch-,4,h/8,w/8
        #context=barch,seq_len,dim
        # time=1,1280
        skip_connection=[]
        for layers in self.encoder:
            x=layers(x,context,time)
            skip_connection.append(x)
        x=self.bottleneck(x,context,time)

        for layers in self.decoders:
            x=torch.cat(x,skip_connection.pop(),dim=1)
            x=layers(x,context,time)
        return x
        

# %%
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x



