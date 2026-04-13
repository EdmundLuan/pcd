"""Wrapper class for performing coupled and convex hull guided diffusion."""

import time
import torch
from torch import nn
import torch.nn.functional as F
try:
    from IPython import get_ipython
    if 'ipykernel' in str(type(get_ipython())):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm


from typing import List
from diffusers import (
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from src.models.sd import UnconditionalStableDiffusion
from src.utils.projection_util import batch_project_onto_convex_hull_mirror_descent
from src.utils.dataset_util import map_tensor_range
from src.utils.device_util import recursive_to
from src.utils.time_util import compute_time_stats
from src.utils.logging_util import LoggingUtils


logger = LoggingUtils.configure_logger(log_name=__name__)

CACHE_DEVICE = "cpu"



class ProjectedCoupledUnconditionalStableDiffusion(nn.Module):
    
    def __init__(
        self,
        M      : List[UnconditionalStableDiffusion] = [],
        D      : callable = None,
        device : torch.device = None,
        **kwargs
    ):
        """
        Wrapper Class for performing coupled diffusion between ddpm models / distributions U & V,
        while using some distance function D(x,y) to enforce coupling during sampling.
        
        Args:
            M (List[UnconditionalStableDiffusion]) : List of UnconditionalStableDiffusion Models to sample X_m ~ U_m(x) from.
            D (callable)  : Distance function D(x,y) to enforce coupling relation. This function should be able to take in:
                             - batch_x : Torch float tensor of shape (bs,n,n), and values in range [0,1].
                             - batch_y : Torch float tensor of shape (bs,n,n), and values in range [0,1].
                            Additional input arguments provided are:
                             - batch_t   : Torch long tensor of shape (bs) or (bs,1), and values in range [0, num_t - 1].
                             - num_t     : Integer representing the number of diffusion timesteps required, i.e. 150.
                            This distance function should output a scalar value representing the expected / mean difference between 'x' and 'y'.
            device (torch.device): Torch device to run the model on (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        
        assert len(M) > 0
        
        self.M = M
        
        assert all(M[0].num_steps == m.num_steps for m in M)
        assert all(M[0].latent_width == m.latent_width for m in M)
        assert all(M[0].latent_height == m.latent_height for m in M)
        assert all(M[0].latent_channels == m.latent_channels for m in M)
        assert all(M[0].dtype == m.dtype for m in M)
        assert all(str(M[0].device) == str(m.device) for m in M)
        assert all(len(M[0].scheduler.timesteps) == len(m.scheduler.timesteps) for m in M)
        assert all(isinstance(m.scheduler, DDPMScheduler) for m in M)
        
        self.dtype = M[0].dtype
        self.device = device
        self.device_type = str(self.device).split(":")[0]
        self.enable_amp = self.dtype != torch.float
        print(f"[Projected Coupled Diffusion] AMP enabled: {self.enable_amp}")
        
        assert D is not None, f"Distance function, D: R^n -> R, to minimise cannot be None!"
        self.D = D
        
        self.to(device)
    
    
    
    def get_distance_func_gradient(self, X: List[torch.Tensor], wrt_idx: int=0, d_args=None):
        """
        Function to compute distance D(x,y) and derive gradients w.r.t. one of the variales.
        
        Args:
            X       : List of Variable x_m
            wrt_idx : Target variable to compute gradients of D(x,y) w.r.t.
            d_args (dict): Optional arguments for distance function D(x,y).
        
        Returns:
            distance_grad: Gradients of D(x,y) w.r.t. one of the variables.
        """
        assert wrt_idx in list(range(len(X))), f"Not Supported Variable Index: {wrt_idx}"
        
        X_cloned = [x.clone().detach() for x in X]  # full shallow list copy + tensor copies
        X_cloned[wrt_idx].requires_grad_(True)
        
        distance = self.D(X_cloned, **d_args)
        distance_grad = torch.autograd.grad(distance, X_cloned[wrt_idx])[0]
        
        return distance_grad
    
    
    
    "Master method for performing projected coupled diffusion."
    def projected_coupled_forward(
        self, 
        num_images, 
        proj_exemplars: List[torch.Tensor], 
        proj_vae_sample_posterior=True,
        proj_skip_last=True, 
        proj_timesteps=[],
        proj_md_num_iters=10000, 
        proj_md_lr=0.01, 
        coup_grad_scale: float = 0.1, 
        enable_projection: bool = True,
        enable_coupling: bool = True,
        verbose=False, 
        prior=torch.randn, 
        decode_intermediates=False, 
        intermediates_idx=[], 
        prompt: str = "", 
        cfg_guidance_scale: float = 0.,
        noise_std_scaling_factor: float = 1.0,
    ):
        """
        Orchestrator function to run coupled diffusion using ddpm models M, along with
        gradients of the distance fucntion D(x1,x2..xN) w.r.t. xi ~ M_i(x).
        
        Core Implementation:
         d1 = Proj [ - nabla_x_1 * D(x1,x2,..xN) + nabla_x1 * m1(x1) ]dt + sqrt(2)*dW1
         d2 = Proj [ - nabla_x_2 * D(x1,x2,..xN) + nabla_x2 * m2(x2) ]dt + sqrt(2)*dW2
         ...
         dN = Proj [ - nabla_x_N * D(x1,x2,..xN) + nabla_xN * mN(xN) ]dt + sqrt(2)*dWN
        
        Args:
            num_images (int):   Number of samples to generate from each ddpm model.
            grad_scale (float): Scaling constant for D(x1,x2..xN) gradient.
        
        Returns:
            samples      : [ x^0_1, x^0_2, ... x^0_N ]
            intermediates: [ [z^T,z^t-1,...,z^0]_1, [z^T,z^t-1,...,z^0]_2, ... [z^T,z^t-1,...,z^0]_N ]
            time_stats   : {...}
        """
        
        if enable_projection:
            assert len(proj_exemplars) == len(self.M), f"Number of provided exemplar groups ({len(proj_exemplars)}) do not match number of coupled models ({len(self.M)})!"
        
        
        for m in self.M:
            m.configure_eval()
        self.eval()
        
        # Storing decoded intermediate latents & time benchmarking
        intermediates = {
            "model_raw"                 : [[] for i,m in enumerate(self.M)],
            "latents"                   : [[] for i,m in enumerate(self.M)],
            "latents_decoded"           : [[] for i,m in enumerate(self.M)],
        }
        time_stats = {
            "diffusion_step_overall" : {"time": []},
            "model"                  : {"time": []},
        }
        if enable_coupling:
            intermediates["latents_coupled"]           = [[] for i,m in enumerate(self.M)]
            intermediates["latents_coupled_decoded"]   = [[] for i,m in enumerate(self.M)]
            time_stats["coupling_overall"] = {"time": []}
        if enable_projection:
            intermediates["latents_projected"]         = [[] for i,m in enumerate(self.M)]
            intermediates["latents_projected_decoded"] = [[] for i,m in enumerate(self.M)]
            intermediates["lambda"]                    = [[] for i,m in enumerate(self.M)]
            time_stats["projection_overall"] = {"time": []}
            time_stats["projection"]         = {"time": []}
        
        with torch.no_grad():
            
            if enable_projection:
                # prep & encode exemplars, [[k,3,H,W], ..] -> [[k,lc,lh,lw], ..]
                proj_exemplars = [e.to(self.device) for e in proj_exemplars]
                proj_exemplar_latents = [m.vae_encode_no_grad(proj_exemplars[i], num_images, proj_vae_sample_posterior) for i,m in enumerate(self.M)]
            
            # (dummy) prompt embeds
            uncond_prompt_embs = [m.get_prompt_embeddings(num_images, prompt=prompt) for i,m in enumerate(self.M)]
            
            # Sample initial latents, z_{T}
            latents = [
                prior(
                    (num_images, m.latent_channels, m.latent_height, m.latent_width),
                    device = self.device
                ) * m.scheduler.init_noise_sigma
                for m in self.M
            ]
            
            timesteps = [m.scheduler.timesteps for i,m in enumerate(self.M)]
            assert all(len(timesteps[0]) == len(timestep) for timestep in timesteps)
        
        
        # Diffusion Loop
        for step_idx in tqdm(range(len(timesteps[0])), desc="Diffusion Steps"):
            
            time_start_overall = time.time()
            
            t = [timestep[step_idx] for timestep in timesteps]
            t_ints = [t[i].item() for i,m in enumerate(self.M)]
            
            
            batch_ts = [
                (t_ints[i] * torch.ones(num_images)).long().to(self.device)
                for i,m in enumerate(self.M)
            ]
            
            
            with torch.no_grad():
                
                time_start_model = time.time()
                # predict noise
                noise_pred = [
                    m.phi(torch.cat([latents[i]]*2, dim=0), torch.cat([batch_ts[i]]*2, dim=0), encoder_hidden_states=uncond_prompt_embs[i])
                    for i,m in enumerate(self.M)
                ]
                time_stats["model"]["time"].append(time.time() - time_start_model)
                
                # classifier-free guidance
                guided = []
                for i,m in enumerate(self.M):
                    u, c = noise_pred[i].chunk(2)
                    cfg_noise = u + cfg_guidance_scale * (c - u)
                    guided.append(cfg_noise)
                
                
                if decode_intermediates:
                    for i,m in enumerate(self.M):
                        to_keep = torch.stack([guide for idx,guide in enumerate(guided[i]) if idx in intermediates_idx])
                        intermediates["model_raw"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
                        
                        to_keep = torch.stack([latent for idx,latent in enumerate(latents[i]) if idx in intermediates_idx])
                        intermediates["latents"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
                        
                        decoded = m.vae_decode_no_grad(latents[i], num_images, use_tqdm=False)
                        to_keep = torch.stack([inter for idx,inter in enumerate(decoded) if idx in intermediates_idx])
                        intermediates["latents_decoded"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
            
            
            if enable_coupling:
                time_start_coupling_overall = time.time()
                # compute gradients of distance function D(x,y) w.r.t. X ~ U(x) & Y ~ V(y)
                # ========================================================================
                # Runs :
                #  d1 = [ - nabla_x_1 * D(x1,x2,..xN) + nabla_x1 * m1(x1) ]dt + sqrt(2)*dW1
                #  d2 = [ - nabla_x_2 * D(x1,x2,..xN) + nabla_x2 * m2(x2) ]dt + sqrt(2)*dW2
                #  ...
                #  dN = [ - nabla_x_N * D(x1,x2,..xN) + nabla_xN * mN(xN) ]dt + sqrt(2)*dWN
                assert all(not torch.isnan(l).any() for l in latents), "Latents contains NaNs!"
                
                with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                    d_args = {
                        "t": batch_ts,    # shape: [[num_images],...]
                        "num_t": [int(m.scheduler.config.num_train_timesteps) for m in self.M]
                    }
                    latents_grad = [
                        self.get_distance_func_gradient(X=latents, wrt_idx=i, d_args=d_args)
                        for i,m in enumerate(self.M)
                    ]
                
                assert all(not torch.isnan(g).any() for g in latents_grad), "Latents Grad contains NaNs!"
                
                
                with torch.no_grad():
                    guided_grad_updated = []
                    for i,m in enumerate(self.M):
                        # alpha_bar per-sample
                        alpha_bar_t_i = m.scheduler.alphas_cumprod.to(batch_ts[i].device)[batch_ts[i]]   # [B]
                        alpha_bar_t_i = alpha_bar_t_i.to(latents_grad[i].device, latents_grad[i].dtype)
                        sigma_t = (1 - alpha_bar_t_i).sqrt().view(-1, 1, 1, 1)           # [B,1,1,1]
                        
                        grad = latents_grad[i]                                           # [B,C,H,W]
                        norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8) # [B,1]
                        unit_grad = grad / norm.view(-1,1,1,1)
                        
                        g = coup_grad_scale * sigma_t * unit_grad                        # [B,C,H,W]
                        guided_i = guided[i] + g
                        
                        print(f"\n[Latent {i}] grad norm: {g.flatten(1).norm(dim=1).mean().item():.5f} | min: {g.min().item():.5f} | max: {g.max().item():.5f} | mean: {g.mean().item():.5f}")
                        
                        guided_grad_updated.append(guided_i)
                    
                    guided = guided_grad_updated
                
                assert all(not torch.isnan(g).any() for g in guided_grad_updated), "Guided Grad Updated contains NaNs!"
                time_stats["coupling_overall"]["time"].append(time.time() - time_start_coupling_overall)
                # ========================================================================
            
            
            with torch.no_grad():
                
                # z_{t} -> z_{t-1}
                # ------------------------------------------------------------------------
                # latents = [
                #     m.scheduler.step(guided[i], t[i], latents[i]).prev_sample
                #     for i,m in enumerate(self.M)
                # ]
                # Implementation in smaller unique-t batches
                tmp_latents = []
                for i,m in enumerate(self.M):
                    latent_i_bucket = torch.empty_like(latents[i])
                    for unique_t in tqdm(torch.unique(batch_ts[i]), desc="Reverse Step"):
                        mask = (batch_ts[i] == unique_t)
                        idx  = mask.nonzero(as_tuple=False).view(-1) # [n_t]
                        
                        out = m.scheduler.step(
                            model_output   = guided[i][idx],
                            timestep       = int(unique_t.item()),
                            sample         = latents[i][idx],
                            variance_scale = noise_std_scaling_factor,
                        ).prev_sample  # [n_t,lc,lh,lw]
                        
                        # Scatter back to the appropriate indexes
                        latent_i_bucket[idx] = out
                    tmp_latents.append(latent_i_bucket)
                latents = tmp_latents
                # ------------------------------------------------------------------------
            
            
            if enable_coupling:
                with torch.no_grad():
                    if decode_intermediates:
                        for i,m in enumerate(self.M):
                            to_keep = torch.stack([latent for idx,latent in enumerate(latents[i]) if idx in intermediates_idx])
                            intermediates["latents_coupled"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
                            
                            decoded = m.vae_decode_no_grad(latents[i], num_images, use_tqdm=False)
                            to_keep = torch.stack([inter for idx,inter in enumerate(decoded) if idx in intermediates_idx])
                            intermediates["latents_coupled_decoded"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
            
            
            if enable_projection:
                # Project z_{t-1} onto exemplars' latents
                # ================================================================
                time_start_projection_overall = time.time()
                
                with torch.no_grad():
                    for i,m in enumerate(self.M):
                        
                        t_int = t_ints[i]
                        
                        if proj_skip_last and (t_int == m.scheduler.timesteps[-1].item()):
                            print(f"Skipping projection on last step: {t_int}")
                        elif t_int not in proj_timesteps:
                            # print(f"Skipping projection on step: {t_int}")
                            pass
                        else:
                            time_start_projection = time.time()
                            
                            assert not torch.isnan(proj_exemplar_latents[i]).any(), f"Exemplars:{i} contains NaNs!"
                            assert not torch.isnan(latents[i]).any(), f"Latents:{i} contains NaNs!"
                            
                            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enable_amp):
                                latents_proj_i, lambd_i = batch_project_onto_convex_hull_mirror_descent(
                                    ext_points    = latents[i],               # [bs,lc,lh,lw]
                                    hull_points   = proj_exemplar_latents[i], # [ k,lc,lh,lw]
                                    learning_rate = proj_md_lr,
                                    num_iter      = proj_md_num_iters,
                                    verbose       = verbose,
                                    return_lambdas= True
                                )
                                if m.rescale_proj_latent_norm:
                                    latents_proj_i = m.scale_L2_norm_exemplars(latents_proj_i, lambd_i, proj_exemplar_latents[i])
                            
                            
                            latents_i = latents_proj_i
                            
                            latents[i] = latents_i
                            
                            assert not torch.isnan(latents[i]).any(), f"Latents:{i} contains NaNs!"
                            
                            time_stats["projection"]["time"].append(time.time() - time_start_projection)
                            
                            with torch.no_grad():
                                to_keep = torch.stack([lamb for idx,lamb in enumerate(lambd_i) if idx in intermediates_idx])
                                intermediates["lambda"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
                
                time_stats["projection_overall"]["time"].append(time.time() - time_start_projection_overall)
                # ================================================================
                
                
                with torch.no_grad():
                    if decode_intermediates:
                        for i,m in enumerate(self.M):
                            to_keep = torch.stack([latent for idx,latent in enumerate(latents[i]) if idx in intermediates_idx])
                            intermediates["latents_projected"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
                            
                            decoded = m.vae_decode_no_grad(latents[i], num_images, use_tqdm=False)
                            to_keep = torch.stack([inter for idx,inter in enumerate(decoded) if idx in intermediates_idx])
                            intermediates["latents_projected_decoded"][i].append(to_keep.detach().clone().to(dtype=self.dtype, device=CACHE_DEVICE))
            
            
            time_stats["diffusion_step_overall"]["time"].append(time.time() - time_start_overall)
        
        
        with torch.no_grad():
            
            # decode latents
            samples = [
                m.vae_decode_no_grad(latents[i], num_images, use_tqdm=False)
                for i,m in enumerate(self.M)
            ]
            
            # denormalize
            samples = [
                m.convert_decoded_latents(samples[i], num_images, clamp=True).detach()
                for i,m in enumerate(self.M)
            ]
            
            if decode_intermediates:
                decoded_list = [
                    "latents_decoded",
                    *(["latents_coupled_decoded"]   if enable_coupling   else []),
                    *(["latents_projected_decoded"] if enable_projection else []),
                ]
                for i,m in enumerate(self.M):
                    for k in decoded_list:
                        intermediates[k][i] = torch.cat(intermediates[k][i], dim=0).to(self.device)
                        intermediates[k][i] = m.convert_decoded_latents(intermediates[k][i], num_images, clamp=True).to(CACHE_DEVICE)
                        intermediates[k][i] = list(torch.split(intermediates[k][i], len(intermediates_idx), dim=0))
            
            
            time_stats = compute_time_stats(time_stats)
        
        samples = recursive_to(samples, to_device=CACHE_DEVICE)
        intermediates = recursive_to(intermediates, to_device=CACHE_DEVICE)
        
        return samples, intermediates, time_stats
