"""Script to perform coupled projected diffusion between multiple diffusion models."""

import os
import time
import json
import math
import torch
import random
import argparse
from PIL import Image
from typing import List
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict

import clip
import lpips
from diffusers import (
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from src.schedulers.custom_ddpm import CustomDDPMScheduler
from src.models.latent_classifier_resnet_enc_multihead import (
    mount_resnet_multi_head_latent_classifier_configs,
    build_time_aware_multi_pred_head_classifier,
)
from src.models.sd import UnconditionalStableDiffusion
from src.models.sd_convex_hull_guided_coupled_diffusion import ProjectedCoupledUnconditionalStableDiffusion
from src.utils.device_util import get_device, set_all_seed, recursive_to, DTYPE_MAP
from src.utils.projection_util import parse_projection_timesteps
from src.utils.dataset_util import map_tensor_range, resize_imgs
from src.utils.visual_util import view_images
from src.utils.model_util import display_num_param
from src.utils.metric_util import (
    compute_clip_similarity_to_exemplars,
    compute_lpips_to_exemplars,
    compute_intra_batch_lpips,
)
from src.utils.logging_util import LoggingUtils
from src.utils.visual_util import (
    plot_interpolated_images, 
)
from src.utils.scheduler_util import SCHEDULER_MAPPING

logger = LoggingUtils.configure_logger(log_name=__name__)




# maximum number of samples to visualise
MAX_NUM_SAMPLES_TO_VISUALISE=10



def main(args):
    """Main orchestrator function."""
    
    if not args.enable_coupling:
        logger.warning("Coupling DISABLED.")
    if not args.enable_projection:
        logger.warning("Projection DISABLED.")
    
    
    # NOTE: auto override to k=1 if MASTER_ENABLE_PROJECTION=False
    # During projection, setting this std to a absurdly high value is somewhat ok because the latents
    # will be confined to only be in the valid convex hull. Without projection, this has to be set to
    # k=1 so we don't interfere with the regular diffusion process!
    if not args.enable_projection:
        args.noise_std_scaling_factor = 1.
    # ================================================
    
    
    # print all args (logging purposes)
    [print(f"[ARGS] {k:<25} : {v}") for k, v in vars(args).items()]
    
    # create output dir
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    # set torch device & random seed
    device = get_device(args.device)
    set_all_seed(seed=args.seed, deterministic=args.deterministic)
    
    # local generator for misc tasks that require randomness
    local_rng = random.Random(args.seed)
    local_generator_cpu = torch.Generator(device="cpu")
    local_generator_cpu.manual_seed(args.seed)
    local_generator_gpu = torch.Generator(device=device)
    local_generator_gpu.manual_seed(args.seed)
    
    
    
    
    # load SD model(s)
    # --------------------------------------------------------------
    net = UnconditionalStableDiffusion(
        args                     = None,
        model_id                 = args.sd_model_id,
        num_steps                = args.sd_num_steps,
        sample_height            = args.sd_sample_height,
        sample_width             = args.sd_sample_width,
        latent_height            = args.sd_latent_height,
        latent_width             = args.sd_latent_width,
        latent_channels          = args.sd_latent_channels,
        model_range              = (args.sd_model_min, args.sd_model_max),
        device                   = device,
        dtype                    = DTYPE_MAP[args.sd_dtype],
        is_text_conditional      = args.sd_is_text_conditional,
        rescale_proj_latent_norm = args.rescale_projected_latent_norm,
        use_custom_ddpm_scheduler=True
    )
    display_num_param(net)
    # --------------------------------------------------------------
    
    
    
    # load Classifier model for coupling
    # --------------------------------------------------------------
    classifier_args = mount_resnet_multi_head_latent_classifier_configs(args.classifier_config, argparse.Namespace())
    [print(f"[CLS] {k:<32} : {v}") for k, v in vars(classifier_args).items() if k not in ['train_images', 'test_images']]
    
    if os.path.exists(args.classifier_weights):
        classifier = build_time_aware_multi_pred_head_classifier(classifier_args, device)
        classifier.load_state_dict(torch.load(args.classifier_weights, map_location=device))
        classifier.eval()
        display_num_param(classifier)
    else:
        raise FileNotFoundError(f"Model weights not found at {args.classifier_weights}. Please train the model first.")
    
    assert args.coupling_var in classifier.heads.keys(), f"Coupling Var: '{args.coupling_var}' unsupported. Must be one of {classifier.heads.keys()}"
    # --------------------------------------------------------------
    
    
    
    
    # filter & load exemplars data
    # --------------------------------------------------------------
    preprocess = transforms.Compose([
        transforms.Resize((args.sd_sample_height, args.sd_sample_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # equiv. to map_tensor_range(x, in_range=(0,1), out_range=(-1,1))
    ])
    
    exemplar_sources = [args.exemplars_a, args.exemplars_b]
    
    # load custom exemplars
    exemplar_classes_str = [
        f"{i}-" + "-".join([str(Path(i).stem) for i in exemplar_src])
        for i, exemplar_src in enumerate(exemplar_sources)
    ]
    
    # load images
    exemplars = [[], []]
    for i, exemplar_src in enumerate(exemplar_sources):
        for img_path in exemplar_src:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = preprocess(img)  # [3,H,W], [-1, 1]
                exemplars[i].append(img)
            else:
                raise Exception(f"Image not found: {img_path}")
        
        exemplars[i] = torch.stack(exemplars[i], dim=0)
        
        print(f"[exemplars] Min: {exemplars[i].min().item():.4f}, Max: {exemplars[i].max().item():.4f}")
        exemplars[i] = map_tensor_range(
            tensor    = exemplars[i],
            in_range  = (-1., 1.),
            out_range = net.model_range
        )
        print(f"[exemplars] Min: {exemplars[i].min().item():.4f}, Max: {exemplars[i].max().item():.4f}")
        logger.info(f"All exemplars data   : {exemplars[i].size()}")    # [N, 3, 512, 512], Tensor
    # --------------------------------------------------------------
    
    
    
    
    # visualize exemplars
    for i in range(len(exemplars)):
        view_images(
            images = map_tensor_range(
                tensor    = exemplars[i],
                in_range  = net.model_range,
                out_range = (0., 1.)
            ), 
            num_cols  = min(10, exemplars[i].size(0)),  # limit to 10 columns
            title     = "Conditioning Exemplars", 
            font_size = 15, 
            save_dpi  = 300,
            output_path=os.path.join(f"{args.outputs_dir}", f"Conditioning_Exemplars_{i}_{exemplar_classes_str[i]}.png")
        )
        
        # save (for reproducibility)
        torch.save(exemplars[i].detach().cpu(), os.path.join(f"{args.outputs_dir}", f"exemplars_condition_{i}_{exemplar_classes_str[i]}.pt"))
    
    
    
    
    # linearly interpolate between exemplars' latents and get VAE Decoder to decode
    # --------------------------------------------------------------
    # [BS,lc,lh,lw]
    exemplar_latents = [
        net.vae_encode_no_grad(exemplars[i].to(net.device), args.num_samples_to_generate, sample_posterior=True).float()
        for i in range(len(exemplars))
    ]
    
    for i in range(len(exemplars)):
        if len(exemplars[i]) == 2:
            z0, z1 = exemplar_latents[i][0], exemplar_latents[i][1]  # Each of shape [lc, lh, lw]
            
            interp_scales = torch.linspace(0.0, 1.0, steps=11).to(net.device)  # 0.0 to 1.0 inclusive, 11 steps
            
            # Collect interpolated decoded images
            decoded_images = []
            for alpha in interp_scales:
                z_interp = (1 - alpha) * z0 + alpha * z1  # Linear interpolation
                z_interp = z_interp.unsqueeze(0)  # Add batch dim: [1, lc, lh, lw]
                
                # Decode the latent to image
                x_interp = net.vae_decode_no_grad(z_interp, 1, use_tqdm=False)  # Shape: [1, C, H, W]
                decoded_images.append(x_interp)
            
            # Stack all interpolated images: [11, C, H, W]
            decoded_images = torch.cat(decoded_images, dim=0)
            decoded_images = map_tensor_range(
                tensor    = decoded_images,
                in_range  = (-1, 1),
                out_range = (0, 1)
            )
            decoded_images = decoded_images.clamp(0, 1)
            decoded_images = recursive_to(decoded_images, to_device="cpu", to_dtype=torch.float32)
            interp_scales = recursive_to(interp_scales, to_device="cpu")
            plot_interpolated_images(
                decoded_images, 
                interp_scales, 
                filename=os.path.join(args.outputs_dir, f"interpolated_images_{i}_{exemplar_classes_str[i]}.png")
            )
    # --------------------------------------------------------------
    
    
    
    
    # compile timesteps to do projection
    # --------------------------------------------------------------
    T = net.scheduler.timesteps[0].item()
    args.projection_timesteps = str(args.projection_timesteps).replace("T", f"{T}")
    t_list = parse_projection_timesteps(args.projection_timesteps, T)
    
    projection_timesteps      = [t.item() for t in net.scheduler.timesteps if t.item() in t_list]
    skip_projection_timesteps = [t.item() for t in net.scheduler.timesteps if t.item() not in t_list]
    print(f"\nPerforming projection on timesteps:\n{projection_timesteps}")
    print(f"\nSkipping projection on timesteps:\n{skip_projection_timesteps}\n")
    # --------------------------------------------------------------
    
    
    
    # compile intermediates to keep
    # NOTE: different from previous scripts, we only keep the intermediates of the selected ones due to the large memory footprint.
    NUM_SAMPLES_TO_VISUALISE = min(MAX_NUM_SAMPLES_TO_VISUALISE, args.num_samples_to_generate)
    intermediates_idx = torch.randperm(args.num_samples_to_generate, generator=local_generator_cpu)
    intermediates_idx = intermediates_idx[:NUM_SAMPLES_TO_VISUALISE].tolist()
    
    
    
    
    
    # SD Guided Sampling
    # ====================================================================================================
    # ====================================================================================================
    
    net.configure_eval()
    time_start = time.time()
    
    
    """Projected Coupled Diffusion"""
    
    method = "projected_coupled_diffusion"
    savedir = os.path.join(args.outputs_dir, method)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    
    
    def rotated_classifier_head_logits_XOR(
        X: List[torch.Tensor], 
        t: List[torch.Tensor], 
        num_t: int,
    ) -> torch.Tensor:
        """
        Computes the average pairwise XOR between gender classification probs
        across all unique pairs of diffusion samples (for each unique combination of 
        diffusion models), after applying a left rotation to the second sample's class 
        probabilities. This essentially enforces the relation of "male-female" if the
        rotation is set to 0.
        
        Args:
            X (List[torch.Tensor]): List of tensors [x1, x2, ..., xN], each of shape [B, C, H, W].
            t (List[torch.Tensor]): List of timestep tensors [t1, t2, ..., tN] each of shape [B].
            num_t (List[int])     : List of integers [T1, T2, ..., TN] each representing the total 
                                    timesteps in each diffusion model.
        
        Returns:
            torch.Tensor: Scalar tensor representing the average L2 distance between all pairs.
        """
        assert all(X[0].shape == x.shape for x in X), "All tensors in X must have the same shape"
        
        total_loss = 0.0
        count = 0
        
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                xi = map_tensor_range(
                    tensor   = X[i],
                    in_range = net.model_range,
                    out_range= (classifier_args.model_min, classifier_args.model_max),
                )  # [B,C,H,W]
                xj = map_tensor_range(
                    tensor   = X[j],
                    in_range = net.model_range,
                    out_range= (classifier_args.model_min, classifier_args.model_max),
                )  # [B,C,H,W]
                ti = t[i]  # [B], normalisation to [0,1] done in classifier itself
                tj = t[j]  # [B]
                logits_i = classifier(xi, ti)[args.coupling_var]  # [B,n_classes]
                logits_j = classifier(xj, tj)[args.coupling_var]  # [B,n_classes]
                
                assert args.coupling_var == "age_group"
                """Collate age groups `0-2` to `40-49` as one representing `young`, and others representing `old`."""
                
                YOUNG_END = 7
                def collapse_logits_logsumexp(logits, young_end=YOUNG_END):
                    young = torch.logsumexp(logits[:, :young_end+1], dim=-1, keepdim=True)
                    old   = torch.logsumexp(logits[:, young_end+1:], dim=-1, keepdim=True)
                    return torch.cat([young, old], dim=-1)  # [B, 2]
                
                logits_i = collapse_logits_logsumexp(logits_i)
                logits_j = collapse_logits_logsumexp(logits_j)
                
                probs_i = F.softmax(logits_i, dim=-1)
                probs_j = F.softmax(logits_j, dim=-1)
                
                # Rotate probs_y (class k → k+n mod N)
                probs_j = torch.roll(probs_j, shifts=args.coupling_fn_rot, dims=-1)
                
                # XOR loss will drive the generatioin into 2 different classes
                loss_xor = 1 - (probs_i * (1 - probs_j) + probs_j * (1 - probs_i))   # [B, C]
                loss_xor = loss_xor.sum(-1).mean()
                
                total_loss += loss_xor
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0).to(device)
    
    
    
    MODELS = [net, net]
    couple = ProjectedCoupledUnconditionalStableDiffusion(
        M = MODELS,
        D = rotated_classifier_head_logits_XOR,
        device = device,
    )
    
    
    # batch_x_0     : List[torch.Tensor], each tensor of shape [B,3,H,W]. List len = num diffusion models coupled.
    # intermediates : List[List[torch.Tensor]], each tensor of shape [B,3,H,W]. Outer List len = num diffusion models coupled. Inner List len = num diffusion steps.
    # time_stats    : Dict
    batch_x_0, intermediates, time_stats = couple.projected_coupled_forward(
        num_images                  = args.num_samples_to_generate,
        enable_coupling             = args.enable_coupling,
        enable_projection           = args.enable_projection,
        proj_exemplars              = [i for i in exemplars],
        proj_vae_sample_posterior   = args.vae_sample_posterior,
        proj_skip_last              = args.projection_skip_last,
        proj_timesteps              = projection_timesteps,
        proj_md_num_iters           = args.md_num_iters,
        proj_md_lr                  = args.md_lr,
        coup_grad_scale             = args.coupling_grad_scale,
        decode_intermediates        = False,
        intermediates_idx           = intermediates_idx,
        prompt                      = args.sd_prompt,
        cfg_guidance_scale          = args.sd_cfg_guidance_scale,
        noise_std_scaling_factor    = args.noise_std_scaling_factor,
    )
    
    # [[bs,3,H,W], ..]  value range: [0., 1.]
    batch_x_0 = recursive_to(batch_x_0, to_device="cpu", to_dtype=torch.float)
    
    # [[T,bs,3,H,W], ..]  value range: [0., 1.]
    intermediates = recursive_to(intermediates, to_device="cpu", to_dtype=torch.float)
    for k in intermediates.keys():
        intermediates[k] = [torch.stack(i, dim=0) for i in intermediates[k]]
    
    
    duration = time.time() - time_start
    
    logger.info(f"Time taken for generation: {duration} seconds")
    logger.info(f"Generated samples: {batch_x_0[0].size()}")
    
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    
    
    # visualize generated samples
    view_images(
        images    = [i for i in batch_x_0],
        num_cols  = int(math.sqrt(args.num_samples_to_generate)),
        title     = "Generated Samples",
        font_size = 15,
        save_dpi  = 300,
        output_path=os.path.join(f"{args.outputs_dir}", method, f"samples.png")
    )
    
    
    # save time stats
    time_stats_sorted = OrderedDict()
    time_stats_sorted["overall"] = duration
    for k, v in time_stats.items():
        time_stats_sorted[k] = {kk: vv for kk, vv in v.items() if kk != "time"}
    time_stats_sorted["time"] = {
        k: v["time"] for k, v in time_stats.items() if "time" in v
    }
    with open(os.path.join(f"{args.outputs_dir}", method, f"time_stats.json"), "w") as f:
        json.dump(time_stats_sorted, f, indent=4)
    
    
    for i in range(len(MODELS)):
        batch_x_0_i = batch_x_0[i]
        
        # save batch_x_0
        torch.save(batch_x_0_i, os.path.join(f"{args.outputs_dir}", method, f"samples_model_{i}_{exemplar_classes_str[i]}.pt"))
        
    del intermediates
    
    # ====================================================================================================
    # ====================================================================================================
    
    
    
    
    
    
    # Metrics / Evaluation
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    
    # load classifier
    img_classifier_args = mount_resnet_multi_head_latent_classifier_configs(args.image_classifier_config, argparse.Namespace())
    img_classifier = build_time_aware_multi_pred_head_classifier(img_classifier_args, device)
    img_classifier.load_state_dict(torch.load(args.image_classifier_weights, map_location=device))
    img_classifier.eval()
    
    # LPIPS
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    
    for i in range(len(MODELS)):
        
        with torch.no_grad():
            batch_x_0_i = batch_x_0[i].to(img_classifier.device)
            
            # already mapped to [0., 1.] during diffusion
            batch_x_0_i_mapped_cls = map_tensor_range(
                tensor    = batch_x_0_i,
                in_range  = (0., 1.),
                out_range = (img_classifier.model_min, img_classifier.model_max),
            )
            exemplars_i_mapped = map_tensor_range(
                tensor    = exemplars[i],
                in_range  = net.model_range,
                out_range = (0., 1.)
            )
            if (batch_x_0_i_mapped_cls.shape[2] == img_classifier.img_height) and \
               (batch_x_0_i_mapped_cls.shape[3] == img_classifier.img_width):
                batch_x_0_i_resized = batch_x_0_i_mapped_cls
            else:
                batch_x_0_i_resized = resize_imgs(
                    x = batch_x_0_i_mapped_cls,
                    m = img_classifier.img_height,
                    mode = 'bicubic'
                )
            
            # 1) Image classifier confidence
            t = torch.zeros((args.num_samples_to_generate,)).float().to(device)
            logits_i = img_classifier(batch_x_0_i_resized, t)
            
            confidence_dict = {"average":{}, "per_sample":{}}
            
            heads = ["age_group", "gender"]
            for head in heads:
                if head in img_classifier.heads:
                    probs = F.softmax(logits_i[head], dim=-1).detach().cpu() # [B, C]
                    classes = getattr(img_classifier, f"{head}_classes")
                    assert len(classes) == probs.shape[1]
                    
                    per_sample = [
                        dict(zip(classes, row.tolist())) for row in probs
                    ]
                    avg = dict(zip(classes, probs.mean(0).tolist()))
                    confidence_dict["average"][head] = avg
                    confidence_dict["per_sample"][head] = per_sample
            
            filename = os.path.join(args.outputs_dir, method, f"metrics_cls_confidence_{i}_{exemplar_classes_str[i]}.json")
            with open(filename, "w") as f:
                json.dump(confidence_dict, f, indent=4)
            
            
            # 2) Intra-batch LPIPS / Perceptual diversity
            intra_lpips = compute_intra_batch_lpips(batch_x_0_i, lpips_fn, device, max_pairs=None)
            
            filename = os.path.join(args.outputs_dir, method, f"metrics_intra_lpips_{i}_{exemplar_classes_str[i]}.json")
            with open(filename, "w") as f:
                json.dump(intra_lpips, f, indent=4)
            
            
            # 3) LPIPS to exemplars (min: distance to nearest exemplar)
            #     - min captures constraint satisfaction
            #     - mean captures how the sample relates to the whole exemplar set
            lpips_proj = compute_lpips_to_exemplars(batch_x_0_i, exemplars_i_mapped, lpips_fn, device, reduce="min")
            
            filename = os.path.join(args.outputs_dir, method, f"metrics_lpips_proj_{i}_{exemplar_classes_str[i]}.json")
            with open(filename, "w") as f:
                json.dump(lpips_proj, f, indent=4)
            
            
            # 4) CLIP cosine similarity to exemplars
            clip_proj = compute_clip_similarity_to_exemplars(batch_x_0_i, exemplars_i_mapped, clip_model, clip_preprocess, device, reduce="mean")
            
            filename = os.path.join(args.outputs_dir, method, f"metrics_clip_proj_{i}_{exemplar_classes_str[i]}.json")
            with open(filename, "w") as f:
                json.dump(clip_proj, f, indent=4)
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    
    
    logger.info("DONE.")





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convex Hull Guided Diffusion")
    
    # CUDA configs
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility. Set -1 to disable setting seed.")
    parser.add_argument("--deterministic", action='store_true', 
                        help="Flag to enable deterministic CUDNN behavior")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to run the models on, i.e. cuda | cuda:0 | cpu")
    # SD Model configs
    parser.add_argument("--sd_model_id", type=str, default="stabilityai/stable-diffusion-2-1-base", 
                        help="Path to LDM yaml config file")
    parser.add_argument("--sd_num_steps", type=int, default=999, 
                        help="Number of diffusion timesteps")
    parser.add_argument("--sd_sample_height", type=int, default=512, 
                        help="Height of generated image sample")
    parser.add_argument("--sd_sample_width", type=int, default=512, 
                        help="Width of generated image sample")
    parser.add_argument("--sd_latent_height", type=int, default=64, 
                        help="Height of latents")
    parser.add_argument("--sd_latent_width", type=int, default=64, 
                        help="Width of latents")
    parser.add_argument("--sd_latent_channels", type=int, default=4, 
                        help="Channel of latents")
    parser.add_argument("--sd_model_min", type=float, default=-1., 
                        help="Model operating value range min")
    parser.add_argument("--sd_model_max", type=float, default=1., 
                        help="Model operating value range max")
    parser.add_argument("--sd_dtype", type=str, default="float32", 
                        help=f"Model operating datatype. Must be one of: {list(DTYPE_MAP.keys())}")
    parser.add_argument("--sd_is_text_conditional", action='store_true', 
                        help="Model type: UNet2DModel or UNet2DConditionModel (text)")
    parser.add_argument("--sd_prompt", type=str, default="",
                        help="Prompt for SD model. Only applied if --sd_is_text_conditional flag is set.")
    parser.add_argument("--sd_cfg_guidance_scale", type=float, default=0.0,
                        help="Classifier Free Guidance scale for --prompt. Only applied if --sd_is_text_conditional flag is set.")
    parser.add_argument("--vae_sample_posterior", action='store_true', 
                        help="Flag to enable usage of (μ + σ·ε) or (μ) when encoding exemplars via VAE")
    
    # Latent classifier configs [for coupling]
    parser.add_argument("--classifier_config", type=str, default="configs/model_configs/MNIST_classifier_base.yaml", 
                        help="Path to Classifier yaml config file")
    parser.add_argument("--classifier_weights", type=str, default="model_weights/classifier_mnist.pth", 
                        help="Path to trained Classifier .pth weights file")
    
    # Exemplar configs
    parser.add_argument("--exemplars_a", nargs="+", default=[], 
                        help="Filepaths to exemplar images for diffusion model A.")
    parser.add_argument("--exemplars_b", nargs="+", default=[], 
                        help="Filepaths to exemplar images for diffusion model B.")
    
    # Projection configs
    parser.add_argument("--enable_projection", action='store_true', 
                        help="Flag to enable projection")
    parser.add_argument("--projection_skip_last", action='store_true', 
                        help="Flag to skip projection in the last diffusion step")
    parser.add_argument("--projection_verbose", action='store_true', 
                        help="Flag to enable verbose result output during projection")
    parser.add_argument("--projection_timesteps", type=str, default="all",
                        help="Range of timesteps to apply projection. Supported: :,all,``, x:, :x, x:y, x")
    # ------------------ Projection method specific args: Mirror Descent
    parser.add_argument("--md_num_iters", type=int, default=10000,
                        help="MD: Number of iterations for mirror descent optimization")
    parser.add_argument("--md_lr", type=float, default=0.01,
                        help="MD: Learning rate for mirror descent exponentiated gradient updates")
    # ------------------ Scale L2 Norm of latents
    parser.add_argument("--rescale_projected_latent_norm", action='store_true', 
                        help="Flag to enable rescaling of L2 Norms of Projected latents to match that of Standard Guassian at the latent's dimension.")
    # ------------------ Scaling Std of noise at every diffusion step with k
    #                    x_{t-1} = mu_t  +  sqrt(\hat{beta}_t) * k * z,   z ~ N(0, I)
    parser.add_argument("--noise_std_scaling_factor", type=float, default=1.0,
                        help="Scale the std of the noise term added at every diffusion timestep to promote diversity in generated samples.")
    
    # Coupling configs
    parser.add_argument("--enable_coupling", action='store_true', 
                        help="Flag to enable coupling")
    parser.add_argument("--coupling_grad_scale", type=float, default=1.0,
                        help="Scale for coupling function gradient.")
    parser.add_argument("--coupling_var", type=str, default="gender",
                        help="Variable to perform coupling on. Must be one of latent classifier's heads.")
    parser.add_argument("--coupling_fn_rot", type=int, default=-1,
                        help="How much to rotate the classifier's logits.")
    
    # Generation configs
    parser.add_argument("--num_samples_to_generate", type=int, default=400,
                        help="Number of samples to generate from the DDPM model")
    parser.add_argument("--outputs_dir", type=str, default="outputs", 
                        help="Directory for storing outputs")
    
    # Image Classifier for analysis
    parser.add_argument("--image_classifier_config", type=str, default="model_weights/FFHGA_classifier_resnet_enc_multihead_age_group_1024_gender_128/FFHQA_classifier_resnet_enc_multihead.yaml",
                        help="Path to YAML config file of trained image classifier.")
    parser.add_argument("--image_classifier_weights", type=str, default="model_weights/FFHGA_classifier_resnet_enc_multihead_age_group_1024_gender_128/checkpoints_classifier/best_classifier.pt",
                        help="Path to weights file of trained image classifier.")
    args = parser.parse_args()
    
    main(args)
