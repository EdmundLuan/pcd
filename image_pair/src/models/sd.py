"""Vanilla Unconditonal Stable Diffusion."""

import math
import time
import torch
from torch import nn
from typing import List
try:
    from IPython import get_ipython
    if 'ipykernel' in str(type(get_ipython())):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm


from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from src.schedulers.custom_ddpm import CustomDDPMScheduler
from src.utils.dataset_util import map_tensor_range
from src.utils.time_util import compute_time_stats
from src.utils.logging_util import LoggingUtils


logger = LoggingUtils.configure_logger(log_name=__name__)




class UnconditionalStableDiffusion(torch.nn.Module):
    
    def __init__(
        self,
        args,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        num_steps: int = 999,
        sample_height: int = 512,
        sample_width: int = 512,
        latent_height: int = 64,
        latent_width: int = 64,
        latent_channels: int = 4,
        model_range: tuple = (-1., 1.),
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        is_text_conditional: bool = True,
        rescale_proj_latent_norm: bool = False,
        use_custom_ddpm_scheduler: bool = False,
    ):
        super().__init__()
        
        if args is not None:
            for k, v in vars(args).items():
                setattr(self, k, v)
        
        self.model_id        = model_id
        self.device          = device
        self.num_steps       = num_steps
        self.model_range     = model_range
        self.sample_height   = sample_height
        self.sample_width    = sample_width
        self.latent_height   = latent_height
        self.latent_width    = latent_width
        self.latent_channels = latent_channels
        
        self.is_text_conditional = is_text_conditional
        self.rescale_proj_latent_norm = rescale_proj_latent_norm
        
        self.dtype = dtype
        self.device_type = str(self.device).split(":")[0]
        self.enable_amp = self.dtype != torch.float
        print(f"[Stable Diffusion] AMP enabled: {self.enable_amp}")
        
        if use_custom_ddpm_scheduler:
            ddpm_scheduler = CustomDDPMScheduler
        else:
            ddpm_scheduler = DDPMScheduler
        
        if self.is_text_conditional:
            print("[Stable Diffusion] Loading Text Conditional Model")
            self.tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.scheduler    = ddpm_scheduler.from_pretrained(model_id, subfolder="scheduler")
            self.vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
            self.unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
        else:
            print("[Stable Diffusion] Loading Unconditional Model")
            self.scheduler    = ddpm_scheduler.from_pretrained(model_id, subfolder="scheduler")
            self.vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
            self.unet         = UNet2DModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
        
        # if xFormers is installed, this is even more efficient:
        # self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler.set_timesteps(self.num_steps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        print(f"[Stable Diffusion] len(scheduler.timesteps): {len(self.scheduler.timesteps)}")
        
        # target norm (gaussian at dimensionality D)
        if self.rescale_proj_latent_norm:
            self.D = self.latent_height * self.latent_width * self.latent_channels
            self.target_norm = math.sqrt(self.D - 1/2)
            print(f"[Stable Diffusion] Target L2 Norm at D={self.D}: {self.target_norm:.5f}")
        
        self.to(device)
    
    
    def vae_encode_no_grad(self, points, batch_size, sample_posterior=True, scale_latents=True, use_tqdm=True):
        """Function to encode exemplars via VAE."""
        
        with torch.no_grad():
            all_encoded = self.vae_encode_with_grad(points, batch_size, sample_posterior, scale_latents, use_tqdm)
        
        return all_encoded
    
    
    def vae_encode_with_grad(self, points, batch_size, sample_posterior=True, scale_latents=True, use_tqdm=True):
        """Function to encode exemplars via VAE."""
        
        chunks = torch.split(points, batch_size) # [[k,3,H,W], ..]
        processed_chunks = []
        iter = tqdm(chunks, desc="VAE Encoding") if use_tqdm else chunks
        for chunk in iter:
            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                encoded = self.vae.encode(chunk).latent_dist
            encoded = encoded.sample() if sample_posterior else encoded.mode()
            if scale_latents:
                encoded = encoded * self.vae.config.scaling_factor
            processed_chunks.append(encoded)
        
        all_encoded = torch.cat(processed_chunks, dim=0) # [K,lc,lh,lw]
        return all_encoded
    
    
    def vae_decode_no_grad(self, points, batch_size, unscale_latents=True, use_tqdm=True):
        """Function to decode exemplars via VAE."""
        
        with torch.no_grad():
            all_decoded = self.vae_decode_with_grad(points, batch_size, unscale_latents, use_tqdm)
        
        return all_decoded
    
    
    def vae_decode_with_grad(self, points, batch_size, unscale_latents=True, use_tqdm=True):
        """Function to decode exemplars via VAE."""
        
        chunks = torch.split(points, batch_size) # [[k,lc,lh,lw], ..]
        processed_chunks = []
        iter = tqdm(chunks, desc="VAE Decoding") if use_tqdm else chunks
        for chunk in iter:
            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                if unscale_latents:
                    decoded = self.vae.decode(chunk / self.vae.config.scaling_factor).sample
                else:
                    decoded = self.vae.decode(chunk).sample
            processed_chunks.append(decoded)
        
        all_decoded = torch.cat(processed_chunks, dim=0) # [K,3,H,W]
        return all_decoded
    
    
    def phi(self, latents, t, encoder_hidden_states):
        """Wrapper function for UNet model."""
        
        with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
            if self.is_text_conditional:
                noise_pred = self.unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            else:
                noise_pred = self.unet(latents, t).sample
        
        return noise_pred
    
    
    def convert_decoded_latents(self, decoded, batch_size, clamp=True, use_tqdm=True):
        """Function to convert decoded latents to numpy / visualisation-ready tensors."""
        
        chunks = torch.split(decoded, batch_size) # [[k,3,h,w], ..]
        processed_chunks = []
        iter = tqdm(chunks, desc="Convertting decoded latents") if use_tqdm else chunks
        for chunk in iter:
            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                processed_chunk = map_tensor_range(
                    tensor    = chunk,
                    in_range  = self.model_range,
                    out_range = (0., 1.),
                )
                if clamp:
                    processed_chunk = processed_chunk.clamp(0, 1)
            processed_chunks.append(processed_chunk)
        
        all_processed = torch.cat(processed_chunks, dim=0) # [K,3,H,W]
        return all_processed
    
    
    def get_prompt_embeddings(self, num_samples, prompt=""):
        """Function to get `num_images` embedded tokens for unconditional diffusion."""
        
        if self.is_text_conditional:
            prompts = [prompt] * num_samples
            conditional_tokens = self.tokenizer(
                prompts, 
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, 
                return_tensors="pt"
            )
            unconditional_tokens = self.tokenizer(
                prompts, 
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                conditional_emb = self.text_encoder(conditional_tokens.input_ids.to(self.device))[0]
                unconditional_emb = self.text_encoder(unconditional_tokens.input_ids.to(self.device))[0]
            
            emb = torch.cat([unconditional_emb, conditional_emb], dim=0)
        else:
            emb = None  # dummy return value (not usesd anyway)
        
        return emb
    
    
    "00_no_projection"
    def backward_process(self, num_images, prior=torch.randn, decode_intermediates=False, intermediates_idx=[]):
        """
        Function to generate `num_images` samples with unconditional latent diffusion.
        """
        
        # Storing decoded intermediate latents & time benchmarking
        intermediates = {
            "model_raw": [],
            "latents": [],
            "latents_decoded": [],
        }
        time_stats = {
            "diffusion_step_overall": {"time": []},
            "model": {"time": []}
        }
        
        
        with torch.no_grad():
            
            # dummy prompt embeds
            uncond_prompt_embs = self.get_prompt_embeddings(num_images, prompt="")
            cfg_guidance_scale = 0.   # disable cfg
            
            # Sample initial latents, z_{T}
            latents = prior(
                (num_images, self.latent_channels, self.latent_height, self.latent_width),
                device = self.device
            ) * self.scheduler.init_noise_sigma
            
            for step_idx, t in enumerate(tqdm(self.scheduler.timesteps, desc="Diffusion Steps")):
                
                time_start_overall = time.time()
                
                # predict noise
                twice = torch.cat([latents] * 2, dim=0)
                noise_pred = self.phi(twice, t, encoder_hidden_states=uncond_prompt_embs)
                time_stats["model"]["time"].append(time.time() - time_start_overall)
                
                # classifier-free guidance
                u, c = noise_pred.chunk(2)
                guided = u + cfg_guidance_scale * (c - u)
                
                # z_{t} -> z_{t-1}
                latents = self.scheduler.step(guided, t, latents).prev_sample
                
                if decode_intermediates:
                    to_keep = torch.stack([guide for idx,guide in enumerate(guided) if idx in intermediates_idx])
                    intermediates["model_raw"].append(to_keep.detach().clone().to(self.dtype))
                    
                    to_keep = torch.stack([latent for idx,latent in enumerate(latents) if idx in intermediates_idx])
                    intermediates["latents"].append(to_keep.detach().clone().to(self.dtype))
                    
                    decoded = self.vae_decode_no_grad(latents, num_images, use_tqdm=False)
                    to_keep = torch.stack([inter for idx,inter in enumerate(decoded) if idx in intermediates_idx])
                    intermediates["latents_decoded"].append(to_keep.detach().clone().to(self.dtype))
                
                time_stats["diffusion_step_overall"]["time"].append(time.time() - time_start_overall)
            
            # decode latents
            samples = self.vae_decode_no_grad(latents, num_images, use_tqdm=False)
            
            
            # denormalize
            samples = self.convert_decoded_latents(samples, num_images, clamp=True).detach()
            
            if decode_intermediates:
                intermediates["latents_decoded"] = torch.cat(intermediates["latents_decoded"], dim=0)
                intermediates["latents_decoded"] = self.convert_decoded_latents(intermediates["latents_decoded"], num_images, clamp=True)
                intermediates["latents_decoded"] = list(torch.split(intermediates["latents_decoded"], len(intermediates_idx), dim=0))
        
        time_stats = compute_time_stats(time_stats)
        
        return samples, intermediates, time_stats
    
    
    def configure_train(self, vae=True, unet=True, text_encoder=True):
        """
        Set training mode and unfreeze all weights.
        """
        self.train()
        self._set_requires_grad(self.vae, vae)
        self._set_requires_grad(self.unet, unet)
        if self.is_text_conditional:
            self._set_requires_grad(self.text_encoder, text_encoder)
    
    
    def configure_eval(self, vae=False, unet=False, text_encoder=False):
        """
        Set evaluation mode and freeze all weights.
        """
        self.eval()
        self._set_requires_grad(self.vae, vae)
        self._set_requires_grad(self.unet, unet)
        if self.is_text_conditional:
            self._set_requires_grad(self.text_encoder, text_encoder)
    
    
    def _set_requires_grad(self, module: nn.Module, requires: bool):
        for p in module.parameters():
            p.requires_grad = requires


    def scale_L2_norm_gaussian(self, latents):
        """
        Scales L2 norm of latents to match target L2 norm.
        Supports any latent shape with batch at dim 0.
        """
        flat = latents.view(latents.size(0), -1)
        norms = torch.norm(flat, dim=1).clamp(min=1e-8)  # [BS]
        
        # reshape for broadcasting: [BS, 1, 1, 1, ...]
        view_shape = [latents.size(0)] + [1] * (latents.ndim - 1)
        latents = (latents / norms.view(*view_shape)) * self.target_norm
        return latents


    def scale_L2_norm_exemplars(self, latents, lambd, exemplar_latents):
        """
        Scales L2 norm of latents to match a target L2 norm formed by 
        lambd-weighted combination of exemplar norms.
        
        Args:
            latents          : [bs, lc, lh, lw]
            lambd            : [bs, k]
            exemplar_latents : [k, lc, lh, lw]
        
        Returns:
            latents_scaled   : [bs, lc, lh, lw]
        """
        bs = latents.size(0)
        k = exemplar_latents.size(0)
        
        min_norm = 1e-8
        
        # compute latent norms
        latent_norms = torch.norm(latents.view(bs, -1), dim=1).clamp(min=min_norm)  # [bs]
        
        # compute exemplar latent norms
        exemplar_norms = torch.norm(exemplar_latents.view(k, -1), dim=1).clamp(min=min_norm)  # [k]
        
        # compute target L2 norm per sample using lambd-weighted exemplar norms
        target_norms = torch.matmul(lambd, exemplar_norms)  # [bs]
        
        # Rescale latents
        view_shape = [bs] + [1] * (latents.ndim - 1)
        latents_scaled = (latents / latent_norms.view(*view_shape)) * target_norms.view(*view_shape)
        
        return latents_scaled
