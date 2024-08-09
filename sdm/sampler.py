import torch 
import numpy as np

class DDPMSampler:
    def __init__(self,genrator:torch.Generator,num_training_steps=1000,beta_start:float=0.00085,beta_end:float=0.0120):

        self.betas=torch.linspace(beta_start**0.5,beta_end**0.5,num_training_steps,dtype= torch.float32)**2
        self.alphas=1.0-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=0)# markov chain rule to predict noise as any time step latent-sqrt(alpha)*latnet,l-alpha
        self.one=torch.tensor(1.0)

        self.generator=genrator



        self.num_train_timesteps=num_training_steps
        self.timesteps=torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())
    def set_inference_timesteps(self,num_inference_steps=50):
        self.num_inference_timesteps=num_inference_steps
        step_ration=self.num_train_timesteps//num_inference_steps
        timesteps=(np.arange(0,self.num_inference_timesteps)*step_ration[::-1].copy().astype(np.int64))
        self.timesteps=torch.from_numpy(timesteps)
    def add_noise(self,original_samples:torch.FloatTensor,timesteps:torch.Tensor)->torch.Tensor:
        # original is latent produce by the ecoder part of model and the we sample for this latent distribution mean +varience*noise the formula for this markow chai
        #going form x0 to x1 time (sqrt(alpha_bar)*latent):this is mean and varience:1-alpha_bar=bata 

        aplha_cumprod=self.alphas_cumprod.to(device=original_samples,dtype=original_samples)
        timesteps=timesteps.to(device=original_samples)

        sqrt_apha_cumpord=aplha_cumprod[timesteps]**0.5
        sqrt_apha_cumpord=sqrt_apha_cumpord.flatten()
        while len(sqrt_apha_cumpord.shape)<len(original_samples.shape):
            sqrt_apha_cumpord.unsqueeze(-1)

        sqrt_one_minus_alpha_prod=(1-aplha_cumprod(timesteps))**0.5
        sqrt_one_minus_alpha_prod=sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape)<len(original_samples.shape):
            sqrt_one_minus_alpha_prod=sqrt_one_minus_alpha_prod.unqueeze(-1)
        noise=torch.randn(original_samples.shape,generator=self.generator,device=original_samples.device,dtype=original_samples.dtype)
        noisy_samples=(sqrt_apha_cumpord*original_samples)+sqrt_one_minus_alpha_prod*noise
        return noise # this is how to add noise in latnet according to equation 4
    
    def step(self,timestep,latent:torch.Tensor,model_out_put:torch.Tensor):
        t=timestep
        prev_t=self._get_previous_timestep(t)
        alpha_prod_t=self.alphas_cumprod(timestep)
        alpha_prod_t_previous=self.alphas_cumprod(prev_t)if prev_t>=0 else self.one
        beta_prod_t=1-alpha_prod_t
        beta_prod_t_previous=1-alpha_prod_t_previous
        current_alpha_t=alpha_prod_t/alpha_prod_t_previous
        current_beta_t=1-current_alpha_t
        # predict latent at x0 using formula 15 from ddpm paper
        pred_original_sample=(latent-beta_prod_t**0.5*model_out_put)/ alpha_prod_t**0.5              # model output is bascily the unet noise prediction
        # formula seven of ddpm paper sqrt(alpha_prod)beta/1-alpha_bar)x0 + sqrt(alpha_t)(1-alpha_bar_previous)/1-alpha_bar)xt
        pred_original_sample_coeff=(alpha_prod_t**0.5*current_beta_t)/beta_prod_t
        current_sample_coeff=current_alpha_t**0.5*beta_prod_t_previous/beta_prod_t
        
        # compute the predicted previous sample mean
        pred_prev_sample=pred_original_sample_coeff*pred_original_sample+current_sample_coeff*latent
        
        # compute the varience
    def _get_variance(self,timestep:int)->torch.Tensor:
        prev_t=self._get_previous_timestep(timestep)
        alpha_prod_t=self.alphas_cumprod(timestep)
        alpha_prod_t_previous=self.alphas_cumprod(prev_t)if prev_t >=0 else self.one
        current_bata=1-alpha_prod_t/alpha_prod_t_previous
        variece=(1-alpha_prod_t_previous)/(1-alpha_prod_t)*current_bata
        variece=torch.clamp(variece,min=1e-20)
        return variece
        # compute using formula 7 of ddpm paper
    
    def _get_previous_timestep(self,timestep:int)->int:
        previous=timestep-(self.num_train_timesteps//self.num_inference_timesteps)
        return previous
    