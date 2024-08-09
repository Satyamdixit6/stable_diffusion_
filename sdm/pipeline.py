import torch
import numpy as np
from tqdm import tqdm 
from ddpm import DDPMSampler

WIDTH=512 # ENCODER PART OF MODEL EXCEPT THE 512*512 IMAGE
HEIGHT=512
LATENTS_WIDTH=WIDTH//8
LATENTS_HEIGHT=HEIGHT//8

def generate(prompt,# given instruction for image geration
             ucond_promt=None,# condition for image genration latest model 
             input_image=None,# none for if text to image latent noise will ramdom genrated
             strength=0.8, # how much fredom to model for image genration
             do_cfg=True, # classifier guidence weight*(condition_promt-uncondition)+uncondition
             sampler_name="ddpm",# model will predicit latent probablity distribution miu and varience*sigma/noise
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,# choice btw cpu or gpu
             idle_device=None,# move the models from gpu to cpu to save memory/free the space in gpu 
             tokenizer=None,): # tokenizer token the word into number
    
    with torch.no_grad():# why torch_no_grad becouse we only inference from the model not doing the training step if want then set grad=True
        if not 0<strength<=1: # how much freedom given to model for image genration as the stength go up noise to input image up and more random image will genrate
            raise ValueError("strenth must be between 0 and 1")
        if idle_device:
            to_idle=lambda x: x.to(idle_device)
        else:
            to_idle=lambda x:x

        generator=torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        clip=models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens=tokenizer.batch_encode_plus(
                [prompt],padding="max_length",max_lenght=77
            ).input_ids
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            cond_context=clip(cond_tokens)# convert into embedding embedidng_dim=768
            uncond_token=tokenizer.batch_encode_plus([uncond_token],padding="max_lenth",max_length=77).input_ids

            uncond_token=torch.tensor(uncond_token,torch.long,device=device)
            uncond_context=clip(uncond_token)# dim seq_len ,768
            context=torch.cat([cond_context,uncond_context]) # that will double the batch  size
        else:
            tokens=tokenizer.batch_encode_plus([prompt],padding="max_len",max_len=77).input_ids
            tokens=torch.tensor(tokens,dtype=torch.long,device=device)
            context=clip(tokens)
        to_idle(clip)
        if sampler_name=="ddpm":
            sampler=DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps) 
        else:
            raise ValueError("unknown sampler")
        latents_shape=(1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
        if input_image:
            encoder=models["encoder"]
            encoder.to(device)

            input_image_tensor=input_image.resize((WIDTH,HEIGHT))

            input_image_tensor=np.array(input_image_tensor)
            input_image_tensor=torch.tensor(input_image_tensor)
            input_image_tensor=rescale(input_image_tensor,(0,255),(-1,1))
            input_image_tensor=input_image_tensor.unsqeeze(0)# add batch dimesion
            input_image_tensor=input_image_tensor.permute(0,3,1,2)# couse input of ecoder except channels the hight,width

            encoder_noise=torch.rand(latents_shape,generator=generator,device=device)# here when multiply with stenderd deviation size should be same for nosis


            latents=encoder(input_image_tensor,encoder)# forward function of encoder 

            sampler.set_strength(strength=strength)

            to_idle(encoder)
        else: # if want to genrate the image from the text
            latents=torch.randn(latents_shape,generator=generator,device=device)

        diffusion=models["diffusion"]
        diffusion.to(device)
        
        timesteps=tqdm(sampler.timesteps)

        for i,timesteps in enumerate(timesteps):
            # 1,320 here 320 is postional embedding just like trasformer
            time_embedding=get_time_embedding(timesteps).to(device)

            model_input=latents
            if do_cfg:
                model_input=model_input.repeat(2,1,1,1)
            model_out=diffusion(model_input,context,time_embedding)
            if do_cfg:
                output_cond,output_uncond=model_out.chunk(2)
                model_out=cfg_scale*(output_cond-output_uncond)+output_uncond
            latents=sampler.step(timesteps,latents,model_out)
        to_idle(diffusion)
        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
    def rescale(x,old_range,new_range,clamp=False):
        old_min,old_max=old_range
        new_min,new_max=new_range
        x-=old_min
        x*(new_max-new_min)/(old_max-old_min)
        x+=new_min
        if clamp:
            x=x.clamp(new_min,new_max)
        return x
    
def get_time_embedding(timestep):
    # shape:160
    freqs=torch.pow(10000,-torch.arange(start=0,end=160,dtype=torch.float32)/160)
    x=torch.tensor([timestep],dtype=torch.float32)[:,None]*freqs[None]
    #shape=1,160*2
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)
