"""Util methods to compute metrics like LPIPS, Intra-sample LPIPS & CLIP Similarity."""

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import resize
from itertools import combinations
import random
import numpy as np

# -----------------------------
# Helpers
# -----------------------------
def to_m11(x):
    """[0,1] -> [-1,1]"""
    return x * 2.0 - 1.0

def batch_resize(x, size, mode='bilinear'):
    if mode == 'bilinear':
        interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    elif mode == 'bicubic':
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC
    return resize(x, size=[size, size], interpolation=interpolation)

def ensure_4d(x):
    if isinstance(x, list):
        x = torch.stack(x, dim=0)
    assert x.dim() == 4, f"Expected 4D tensor [B,3,H,W], got {x.shape}"
    return x

def _stats_from_vec(v: torch.Tensor):
    return {
        "mean": v.mean().item(),
        "std": v.std(unbiased=False).item(),
        "min": v.min().item(),
        "max": v.max().item()
    }

# -----------------------------
# 1) Intra-batch LPIPS diversity
# -----------------------------
def compute_intra_batch_lpips(
    batch_imgs_01: torch.Tensor,
    lpips_fn,
    device: torch.device,
    max_pairs: int | None = 2000,
):
    """
    Computes LPIPS across pairs within a batch (diversity).
    batch_imgs_01: [B,3,H,W] in [0,1]
    lpips_fn: an instance of lpips.LPIPS
    max_pairs: if not None, randomly subsample up to this many pairs to save time
    """
    x = ensure_4d(batch_imgs_01).to(device)
    x = to_m11(x)  # LPIPS expects [-1,1]
    B = x.shape[0]

    all_pairs = list(combinations(range(B), 2))
    if max_pairs is not None and len(all_pairs) > max_pairs:
        all_pairs = random.sample(all_pairs, k=max_pairs)

    scores = []
    with torch.no_grad():
        for i, j in all_pairs:
            s = lpips_fn(x[i:i+1], x[j:j+1]).item()
            scores.append(s)

    scores_t = torch.tensor(scores, dtype=torch.float32)
    return {
        "num_pairs": len(all_pairs),
        "stats": _stats_from_vec(scores_t),
        "per_pair": scores  # beware of size, drop if too large
    }

# -----------------------------
# 2) LPIPS to exemplars
# -----------------------------
def compute_lpips_to_exemplars(
    batch_imgs_01: torch.Tensor,
    exemplar_imgs_01: torch.Tensor,
    lpips_fn,
    device: torch.device,
    reduce: str = "min",  # 'min' | 'mean' | 'topk_mean'
    topk: int = 1):
    """
    For each generated sample, compute LPIPS to each exemplar, then aggregate.
    reduce='min': use the minimum LPIPS to any exemplar (strict constraint satisfaction proxy)
    reduce='mean': mean over exemplars
    reduce='topk_mean': mean over the top-k closest exemplars
    """
    x = ensure_4d(batch_imgs_01).to(device)
    e = ensure_4d(exemplar_imgs_01).to(device)

    x = to_m11(x)
    e = to_m11(e)

    B, E = x.shape[0], e.shape[0]
    per_sample_scores = []

    with torch.no_grad():
        # Compute in chunks to save memory if needed
        for i in range(B):
            # Compute LPIPS to all exemplars for sample i
            scores_ie = []
            for j in range(E):
                s = lpips_fn(x[i:i+1], e[j:j+1]).item()
                scores_ie.append(s)
            scores_ie = torch.tensor(scores_ie, device='cpu')

            if reduce == "min":
                agg = scores_ie.min().item()
            elif reduce == "mean":
                agg = scores_ie.mean().item()
            elif reduce == "topk_mean":
                k = min(topk, len(scores_ie))
                agg = torch.topk(scores_ie, k=k, largest=False).values.mean().item()
            else:
                raise ValueError(f"Unknown reduce: {reduce}")

            per_sample_scores.append({
                "all": scores_ie.tolist(),
                "reduced": agg
            })

    reduced_vals = torch.tensor([d["reduced"] for d in per_sample_scores])
    return {
        "reduce": reduce,
        "topk": topk if reduce == "topk_mean" else None,
        "stats": _stats_from_vec(reduced_vals),
        "per_sample": per_sample_scores
    }

# -----------------------------
# 3) CLIP cosine sim to exemplars
# -----------------------------
def compute_clip_similarity_to_exemplars(
    batch_imgs_01: torch.Tensor,
    exemplar_imgs_01: torch.Tensor,
    clip_model,
    clip_preprocess,  # from clip.load(...) if using openai/clip
    device: torch.device,
    reduce: str = "max",  # 'max' | 'mean' | 'topk_mean'
    topk: int = 1,
    normalize: bool = True
):
    """
    Computes cosine similarity between CLIP embeddings of generated samples and exemplars.
    """
    import torchvision.transforms as T

    x = ensure_4d(batch_imgs_01).to(device)
    e = ensure_4d(exemplar_imgs_01).to(device)

    # Use CLIP's preprocess if given, else approximate with standard ImageNet normalization
    def preprocess_clip_like(t):
        # If we pass clip_preprocess (from clip.load), better to use it on PIL Images.
        # This is a tensor-only implementation: resize + normalize to CLIP stats.
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=t.device)[None, :, None, None]
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=t.device)[None, :, None, None]
        # CLIP default input size is 224x224 for ViT-B models
        t = F.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
        t = (t - mean) / std
        return t

    with torch.no_grad():
        x_p = preprocess_clip_like(x)
        e_p = preprocess_clip_like(e)

        x_feats = clip_model.encode_image(x_p).float()
        e_feats = clip_model.encode_image(e_p).float()

        if normalize:
            x_feats = F.normalize(x_feats, dim=-1)
            e_feats = F.normalize(e_feats, dim=-1)

        sims = x_feats @ e_feats.t()  # [B, E]
        # reduce
        per_sample = []
        for i in range(sims.shape[0]):
            s_i = sims[i]
            if reduce == "max":
                agg = s_i.max().item()
            elif reduce == "mean":
                agg = s_i.mean().item()
            elif reduce == "topk_mean":
                k = min(topk, s_i.numel())
                agg = torch.topk(s_i, k=k, largest=True).values.mean().item()
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
            per_sample.append({
                "all": s_i.detach().cpu().tolist(),
                "reduced": agg
            })

    reduced_vals = torch.tensor([d["reduced"] for d in per_sample])
    return {
        "reduce": reduce,
        "topk": topk if reduce == "topk_mean" else None,
        "stats": _stats_from_vec(reduced_vals),
        "per_sample": per_sample
    }
