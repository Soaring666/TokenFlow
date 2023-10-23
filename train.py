import torch
import argparse
import logging
import os
import random
import yaml

import diffusers
import transformers
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
from util import save_video

from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from tokenflow_utils import *
from run_tokenflow_pnp import TokenFlow


logger = get_logger(__name__, log_level="INFO")
device = "cuda"


class MyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        n_frames: int = 40,
    ):
        self.data_path = data_path
        self.n_frames = n_frames
        # load frames
        paths = [os.path.join(self.data_path, "%05d.jpg" % idx) for idx in
                               range(self.n_frames)]
        if not os.path.exists(paths[0]):
            paths = [os.path.join(self.data_path, "%05d.png" % idx) for idx in
                                   range(self.n_frames)]
        frames = [Image.open(paths[idx]).convert('RGB') for idx in range(self.n_frames)]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        self.frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16)
    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, index):
        return self.frames[index]

def Mytest():
    train_dataset = MyDataset(data_path='data/cxk')
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8
    )
    batch_list = []
    for i, batch in enumerate(train_dataloader):
        batch_list.append(batch)
        print(batch.shape)              #(8, 3, 512, 512)
    print(train_dataset[1].shape)       #(3, 512, 512)
    print(len(train_dataset))           #40
    batch_all = torch.cat(batch_list, dim=0)
    print(len(batch_list))
    print(batch_all.shape)               #(40, 3, 512, 512)



@torch.no_grad()
def encode_imgs(imgs, vae, batch_size=10, deterministic=True):
    imgs = 2 * imgs - 1
    latents = []
    for i in range(0, len(imgs), batch_size):
        posterior = vae.encode(imgs[i:i + batch_size]).latent_dist
        latent = posterior.mean if deterministic else posterior.sample()
        latents.append(latent * 0.18215)
    latents = torch.cat(latents)
    return latents           

@torch.no_grad()
def decode_latents(latents, vae, batch_size=10):
    latents = 1 / 0.18215 * latents
    imgs = []
    for i in range(0, len(latents), batch_size):
        imgs.append(vae.decode(latents[i:i + batch_size]).sample)
    imgs = torch.cat(imgs)
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs          

@torch.no_grad()
def get_text_embeds(tokenizer, text_encoder, negative_prompt, prompt, batch_size=1):
    # Tokenize text and get embeddings
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Do the same for unconditional embeddings
    uncond_input = tokenizer(negative_prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                    return_tensors='pt')

    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)      #(2, 77, embedding_dim)
    return text_embeddings

@torch.no_grad()
def ddim_inversion(model, cond, latent_frames, batch_size):
    timesteps = reversed(model.scheduler.timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent_frames.shape[0], batch_size):
            x_batch = latent_frames[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                                                                
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else model.scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = model.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
            latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps

    return latent_frames

#origin ddim sample without cond image
@torch.no_grad()
def ddim_sample_origin(model, x, cond, batch_size):
    timesteps = model.scheduler.timesteps
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, x.shape[0], batch_size):
            x_batch = x[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
            
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else model.scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = model.unet(model_input, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (x_batch - sigma * eps) / mu
            x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
    return x

#ddim sample using cond image
@torch.no_grad()
def ddim_sample_condimage(train_model, model, x, cond, train_dataset, batch_size, new_ref_path=None):
    timesteps = model.scheduler.timesteps

    if new_ref_path is None:
    #get ref image latent
        ref_idx = random.randint(0, len(train_dataset)-1)
        ref_image = train_dataset[ref_idx].to(model.device)         #(3, 512, 512)
        latent_ref = encode_imgs(ref_image.unsqueeze(0).to(torch.float16), model.vae)        #(1, 4, 64, 64)
        inverted_ref = ddim_inversion(model, cond, latent_ref, batch_size=1)
    else:
        ref_image = Image.open(new_ref_path).convert('RGB')
        ref_image = ref_image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        ref_image = T.ToTensor()(ref_image)
        ref_image = ref_image.to(device)
        latent_ref = encode_imgs(ref_image.unsqueeze(0).to(torch.float16), model.vae)        #(1, 4, 64, 64)
        inverted_ref = ddim_inversion(model, cond, latent_ref, batch_size=1)

    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, x.shape[0], batch_size):
            x_batch = x[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
            
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else model.scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            #process downsample feature
            cond_feature = train_model(model, inverted_ref, model_input, t, cond)

            #inject feature
            pnp_cond_t = int(model.config["n_timesteps"] * model.config["pnp_cond_t"])
            model.cond_add_timesteps = model.scheduler.timesteps[:pnp_cond_t]
            register_conv_add(model, model.cond_add_timesteps, cond_feature)

            eps = model.unet(model_input, t, encoder_hidden_states=cond_batch).sample

            #recover the forward process
            register_conv_origin(model)

            pred_x0 = (x_batch - sigma * eps) / mu
            x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
    return x

def train(
    model_config: str,
    output_dir: str = "train_feature_mix/man",
    prompt: str = "man",
    negative_prompt: str = "",
    batch_size: int = 5,
    seed: int = 42,
    learning_rate: int = 3e-5,
    lr_scheduler: str = "constant",
    train_epoch: int = 200,
):

    os.makedirs(output_dir, exist_ok=True)

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     mixed_precision=mixed_precision,
    # )

    # Make one log on every process with the configuration for debugging.

    # logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     transformers.utils.logging.set_verbosity_warning()
    #     diffusers.utils.logging.set_verbosity_info()
    # else:
    #     transformers.utils.logging.set_verbosity_error()
    #     diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
    
    #prepare model and froze grad   
    with open(model_config, "r") as f:
        model_config = yaml.safe_load(f)
    model = TokenFlow(model_config)
    train_model = Feature_process(batch_size)

    train_model.requires_grad = True
    model.unet.requires_grad_(True)
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    #set learning parameters
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        train_model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    # Learning Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_training_steps=train_epoch,
    )

    train_dataset = MyDataset(data_path='data/cxk')
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size
    )
    text_embeds = get_text_embeds(model.tokenizer, model.text_encoder, negative_prompt, prompt)[1].unsqueeze(0)
    # train_model, train_dataloader, optimizer, lr_scheduler, model = accelerator.prepare(
    #     train_model, train_dataloader, optimizer, lr_scheduler, model
    # )

    # Move models to gpu and cast to weight_dtype
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     model.to(accelerator.device, dtype=weight_dtype)
    #     train_model.to(accelerator.device, dtype=weight_dtype)
    # else:
    #     model.to(accelerator.device, dtype=torch.float32)
    #     train_model.to(accelerator.device, dtype=torch.float32)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_epoch}")
    first_epoch = 0
    
    progress_bar = tqdm(range(train_epoch))
    progress_bar.set_description("steps")
    for epoch in range(train_epoch):
        train_model.to(device)
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            # with accelerator.accumulate(train_model):
            ref_idx = random.randint(0, len(train_dataset)-1)
            ##每次都要用vae重新编码，可能效率有点低，可以改进
            ref_image = train_dataset[ref_idx].to(device)         #(3, 512, 512)
            batch = batch.to(device)                #(8, 3, 512, 512)
            latent_cond = encode_imgs(ref_image.unsqueeze(0), model.vae)        #(1, 4, 64, 64)
            latent_batch = encode_imgs(batch, model.vae)            #(8, 4, 64, 64)
            noise = torch.randn_like(latent_cond)

            # Sample a random timestep
            timesteps = torch.randint(0, model.scheduler.num_train_timesteps, (1,), device=device)
            timesteps = timesteps.long()

            noisy_cond = model.scheduler.add_noise(latent_cond, noise, timesteps)
            noisy_batch = model.scheduler.add_noise(latent_batch, noise.repeat(latent_batch.shape[0], 1, 1, 1),
                                                    timesteps.repeat(latent_batch.shape[0]))

            #get downsample feature
            cond_feature = train_model(model, noisy_cond, noisy_batch, timesteps, text_embeds)
            # cond_feature = train_model()

            #inject feature
            pnp_cond_t = int(model.config["n_timesteps"] * model.config["pnp_cond_t"])
            model.cond_add_timesteps = model.scheduler.timesteps[:pnp_cond_t]
            register_conv_add(model, model.cond_add_timesteps, cond_feature)

            # apply the denoising network
            noise_batch_pred = model.unet(noisy_batch, timesteps.repeat(batch.shape[0]), 
                                            encoder_hidden_states=text_embeds.repeat(batch.shape[0], 1, 1))['sample']
            
            #recover the forward process
            register_conv_origin(model)

            # loss = F.mse_loss(cond_feature, torch.zeros_like(cond_feature), reduction="mean")
            loss = F.mse_loss(noise_batch_pred, noise.repeat(batch.shape[0], 1, 1, 1), reduction="mean")
            # accelerator.backward(loss)
            with torch.no_grad():
                loss.backward()
            optimizer.step()
            lr_scheduler.step() 
            optimizer.zero_grad()
            train_loss += loss.item()

        progress_bar.update(1)
        logs = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    #save checkpoint
    save_path = os.path.join(output_dir, f"checkpoint-{train_epoch}")
    torch.save((train_model.state_dict()), save_path)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")

    # #validation process
    # register_conv_origin(model)
    
    # #get latent
    # batch_list = []
    # for i, batch in enumerate(train_dataloader):
    #     batch_list.append(batch)
    # batch_all = torch.cat(batch_list, dim=0).to(device)
    # latent_all = encode_imgs(batch_all, model.vae)            #(40, 4, 64, 64)

    # logger.info("*****running validation of ref image from origin video*****")
    # inv_embed = get_text_embeds(model.tokenizer, model.text_encoder, negative_prompt, prompt)[0]
    # inverted_x = ddim_inversion(model, inv_embed, latent_all, batch_size=8)
    # latent_reconstruction = ddim_sample_condimage(train_model, model, inverted_x, inv_embed, batch_size=8)
    # latent_reconstruction = torch.cat(latent_reconstruction)
    # decoded_latents = decode_latents(latent_reconstruction)
    # for i in range(len(latent_reconstruction)):
    #     T.ToPILImage()(decoded_latents[i]).save(f'{output_dir}/img_ode/%05d.png' % i)
    # save_video(decode_latents, f'{output_dir}/validation_out_fps_10.mp4')

    # logger.info("*****running validation of new ref image*****")

def validate(
    model_config: str,
    model_path: str,
    output_dir: str = "train_feature_mix/man",
    prompt: str = "man",
    negative_prompt: str = "",
    batch_size: int = 5,
    ref_path: str = "data/wt.png",
    seed: int = 42,
    ):

    logger.info("***** running validation *****")
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        set_seed(seed)

    with open(model_config, "r") as f:
        model_config = yaml.safe_load(f)
    model = TokenFlow(model_config)
    train_model = Feature_process(batch_size)
    train_model.load_state_dict(torch.load(os.path.join(model_path, f"checkpoint-200")))
    train_model = train_model.to(device)

    
    train_dataset = MyDataset(data_path='data/cxk')
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size
    )
    #validation process
    register_conv_origin(model)
    with torch.no_grad():
        text_embeds = get_text_embeds(model.tokenizer, model.text_encoder, negative_prompt, prompt)[1].unsqueeze(0)
    
        #get latent
        batch_list = []
        for i, batch in enumerate(train_dataloader):
            batch_list.append(batch)
        batch_all = torch.cat(batch_list, dim=0).to(device)
        latent_all = encode_imgs(batch_all, model.vae)            #(40, 4, 64, 64)

        inv_embed = get_text_embeds(model.tokenizer, model.text_encoder, negative_prompt, prompt)[0].unsqueeze(0).to(device)
        inverted_x = ddim_inversion(model, inv_embed, latent_all, batch_size=batch_size)
        latent_reconstruction = ddim_sample_condimage(train_model, model, inverted_x, inv_embed, train_dataset, batch_size=batch_size, new_ref_path=ref_path)
        decoded_latents = decode_latents(latent_reconstruction, model.vae)
        if ref_path is None:
            os.makedirs(f'{output_dir}/ref_origin', exist_ok=True)
            for i in range(len(latent_reconstruction)):
                T.ToPILImage()(decoded_latents[i]).save(f'{output_dir}/ref_origin/%05d.png' % i)
            save_video(decoded_latents, f'{output_dir}/ref_origin/validation_out_fps_10.mp4')
        else:
            os.makedirs(f'{output_dir}/ref_cond', exist_ok=True)
            for i in range(len(latent_reconstruction)):
                T.ToPILImage()(decoded_latents[i]).save(f'{output_dir}/ref_cond/%05d.png' % i)
            save_video(decoded_latents, f'{output_dir}/ref_cond/validation_out_fps_10.mp4')

    logger.info("***** sampling done! *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_config", type=str, default="configs/config_train.yaml")
    parser.add_argument("--validate_config", type=str, default="configs/config_validation.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger('my_logger')

    if args.train:
        logger.info("******* train process ********")
        config = OmegaConf.load(args.train_config)
        train(**config)
    else:
        logger.info("******* validation process ********")
        config = OmegaConf.load(args.validate_config)
        validate(**config)
    # Mytest()