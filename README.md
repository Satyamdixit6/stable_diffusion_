# stable_diffusion_correct
To train the Stable Diffusion v1-5 model, you should download the weights from the Hugging Face repository. Specifically, you have two options for the weights:

v1-5-pruned-emaonly.ckpt (4.27GB): This checkpoint contains only the Exponential Moving Average (EMA) weights. It uses less VRAM and is suitable for inference.
v1-5-pruned.ckpt (7.7GB): This checkpoint contains both EMA and non-EMA weights. It uses more VRAM and is suitable for fine-tuning1.
Why Choose v1-5-pruned.ckpt for Training?
EMA and Non-EMA Weights: The v1-5-pruned.ckpt includes both EMA and non-EMA weights, which are beneficial for training. EMA weights help in stabilizing the training process and often lead to better generalization.
Fine-Tuning: This checkpoint is specifically designed for fine-tuning, making it ideal for training tasks where you want to adapt the model to a specific dataset or task.
# for speed up the training further use kv_cache in attention i still figurig out this method 
#use of quantization aware model 
