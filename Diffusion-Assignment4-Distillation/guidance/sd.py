from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for SDS
        
        n_latents = latents.shape[0]
        t = torch.randint(self.min_step, self.max_step + 1, 
                          (n_latents,), 
                          dtype=torch.long, 
                          device=self.device)
        noise = torch.randn_like(latents)
        
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        latent_noised = latents * alpha.sqrt() + noise * (1 - alpha).sqrt()
        
        noise_pred = self.get_noise_preds(latent_noised, t, text_embeddings, guidance_scale)

        # grad L_sds
        gradient = grad_scale * (1 - self.alphas[t]) * (noise_pred - noise) 
        gradient = torch.nan_to_num(gradient)

        # latent updated
        # gradient of g is identity
        latent_updated = (latents - gradient).detach()
        
        loss = torch.nn.functional.mse_loss(latents, latent_updated, reduction='mean') / 2

        return loss
        
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        assert src_latents.shape[0] == tgt_latents.shape[0]
        n_latents = src_latents.shape[0]
        
        timesteps = reversed(self.scheduler.timesteps)
        t = torch.randint(self.min_step, self.max_step, 
                        (n_latents,), 
                        dtype=torch.long, 
                        device='cpu')
        t_prev = t-1
        
        t = timesteps[t].cpu()
        t_prev = timesteps[t_prev].cpu()
        
        device = src_latents.device
        noise = torch.randn_like(src_latents).to(device)
        noise_prev = torch.randn_like(src_latents).to(device)
        
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.alphas[t].to(device)
        alpha_bar_t_prev = self.alphas[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)
        
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            
        res = {}
        
        t = t.to(device)
        for name, latent, text_embedding in zip(['src', 'tgt'], [src_latents, tgt_latents], [src_text_embedding, tgt_text_embedding]):
            latent_noised = latent * alpha_bar_t.sqrt() + noise * (1 - alpha_bar_t).sqrt()
            latent_noised_prev = latent * alpha_bar_t_prev.sqrt() + noise_prev * (1 - alpha_bar_t_prev).sqrt()
            noise_pred = self.get_noise_preds(latent_noised, t, text_embedding, guidance_scale)
            
            # Compute x0 prediction
            pred_x0 = (latent_noised - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Compute the posterior mean
            mu = c0 * pred_x0 + c1 * latent_noised

            # Compute zt
            zt = (latent_noised_prev - mu) / sigma_t
            res[name] = zt

        # Compute the gradient and the loss
        grad = res['tgt'] - res['src']
        grad = torch.nan_to_num(grad)

        target = (tgt_latents - grad).detach()
        loss = grad_scale * torch.nn.functional.mse_loss(tgt_latents, target, reduction='mean') / 2

        return loss
    # python main.py --prompt "A red bus driving on a desert road" --loss_type pds --guidance_scale 7.5 --edit_prompt "A yellow school bus driving on a desert road" --src_img_path ./data/imgs/A_red_bus_driving_on_a_desert_road.png
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
