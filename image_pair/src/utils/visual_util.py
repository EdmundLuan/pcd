"""Plotting & Visualisation utility functions."""

import os
import math
import uuid
import torch
import shutil
import imageio
import numpy as np
import seaborn as sns
from typing import List, Union
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

try:
    from IPython import get_ipython
    if 'ipykernel' in str(type(get_ipython())):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm


from src.utils.logging_util import LoggingUtils

logger = LoggingUtils.configure_logger(log_name=__name__)




def tensors_to_gif(
    tensor_list : Union[List[torch.Tensor], List[List[torch.Tensor]]], 
    gif_path    : str = "outputs/output.gif", 
    custom_text : str = "", 
    fps         : int = 10,
    value_range : tuple = (0., 1.)
):
    """
    Converts a list of (B, H, W) or (B, 3, H, W) tensors into a GIF.
    
    If input is a list of tensors, each frame will be a single row.
    If input is a list of list of tensors, each sublist forms a row in the frame.
    
    Args:
        tensor_list (List[Tensor] or List[List[Tensor]]): 
            Tensors of shape (B, H, W) or (B, 3, H, W) or nested list.
        gif_path (str): Path to save the resulting GIF.
        custom_text (str): Text to append to each frame’s title.
        fps (int): Frames per second.
        value_range (tuple): Min/max values for visualization.
    """
    gif_dir = os.path.dirname(os.path.abspath(gif_path))
    os.makedirs(gif_dir, exist_ok=True)
    
    # Create tmp directory inside gif_dir
    tmp_dir = os.path.join(gif_dir, f"tmp_frames_{uuid.uuid4()}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    is_nested = isinstance(tensor_list[0], list)
    num_frames = len(tensor_list)
    
    for i, frame_data in enumerate(tqdm(tensor_list, desc="Saving Frames")):
        rows = frame_data if is_nested else [frame_data]  # Always a list of lists
        num_rows = len(rows)
        num_cols = rows[0].shape[0]
        
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=(num_cols * 2, num_rows * 2)
        )
        if num_rows == 1:
            axes = [axes]
        if num_cols == 1:
            axes = [[ax] for ax in axes]  # Make it 2D list-like for consistency
        
        for row_idx, tensor in enumerate(rows):
            tensor = tensor.detach().cpu()
            
            # ----------------------------
            # Detect whether this is grayscale ([bs,H,W]) or RGB ([bs,3,H,W])
            if tensor.ndim == 3:
                # grayscale: make it [bs, H, W, 1] so we can index uniformly
                tensor = tensor.unsqueeze(1)       # now [B, 1, H, W]
                is_color = False
            elif tensor.ndim == 4 and tensor.shape[1] == 3:
                # RGB
                is_color = True
            else:
                raise ValueError(f"Unsupported tensor shape {tensor.shape}")
            
            # If color, move channels last now so both cases end up as [bs, H, W, C]
            tensor = tensor.permute(0, 2, 3, 1)   # [bs, H, W, 3] / [bs, H, W, 1]
            # ----------------------------
            
            # ----------------------------
            for col_idx in range(num_cols):
                img = tensor[col_idx]  # [H, W, C]
                ax = axes[row_idx][col_idx]
                if is_color:
                    ax.imshow(img, vmin=value_range[0], vmax=value_range[1])
                else:
                    ax.imshow(img[..., 0], cmap="gray", vmin=value_range[0], vmax=value_range[1])
                ax.axis("off")
            # ----------------------------
            # for j in range(bs):
            #     axes[j].imshow(tensor[j], cmap="gray", vmin=value_range[0], vmax=value_range[1])
            #     axes[j].axis("off")
        
        t_val = num_frames - (i + 1)
        fig.suptitle(f"{custom_text} t={t_val}", fontsize=20)
        
        frame_path = os.path.join(tmp_dir, f"frame_{i:09d}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Reserve more vertical space for the suptitle
        plt.savefig(frame_path)
        plt.close()
    
    # Assemble GIF
    images = [imageio.imread(os.path.join(tmp_dir, f"frame_{i:09d}.png")) for i in range(num_frames)]
    imageio.mimsave(gif_path, images, fps=fps)
    
    # Clean up
    shutil.rmtree(tmp_dir)
    print(f"GIF saved to: {gif_path}")




def view_images(
    images      : Union[torch.Tensor, List[torch.Tensor]],
    num_cols    : int = 10,
    cmap        : str = "gray",
    title       : str = None,
    font_size   : int = 15,
    save_dpi    : int = 300,
    output_path : str = "image.png"
):
    """
    Visualizes and saves a grid of images from a single tensor or a list of tensors.
    
    If a list of tensors is provided, each row will show one image from each tensor 
    (i.e., a paired view). All tensors in the list must have the same shape.
    
    Args:
        images (Tensor or List[Tensor]): Batch of images or a list of batches to compare.
        num_cols (int): Max number of columns. Ignored if input is List[Tensor].
        cmap (str): Colormap for grayscale images.
        title (str): Optional title to add to the figure.
        font_size (int): Font size for title.
        output_path (str): Where to save the final plot.
    """
    is_list = isinstance(images, list)
    
    if is_list:
        assert all(isinstance(img, torch.Tensor) for img in images)
        assert all(img.shape == images[0].shape for img in images)
        
        B = len(images)             # number of tensors = rows
        K = images[0].shape[0]     # batch size    = columns
        
        ndim = images[0].ndim
        if ndim == 3:
            is_color = False
            images_np = [
                img.unsqueeze(1).permute(0,2,3,1).detach().cpu().numpy()
                for img in images
            ]
        elif ndim == 4 and images[0].shape[1] == 3:
            is_color = True
            images_np = [
                img.permute(0,2,3,1).detach().cpu().numpy()
                for img in images
            ]
        else:
            raise ValueError(f"Unsupported shape {images[0].shape}")
        
        fig, axes = plt.subplots(B, K, figsize=(K * 2, B * 2))
        
        # Normalize axes layout so `axes[row][col]` is always valid
        if B == 1: axes = [axes]
        if K == 1: axes = [[ax] for ax in axes]
        
        # ** Corrected indexing here **
        for row in range(B):
            for col in range(K):
                ax = axes[row][col]
                img = images_np[row][col]  # <-- row then col
                if is_color:
                    ax.imshow(img)
                else:
                    ax.imshow(img[..., 0], cmap=cmap)
                ax.axis("off")
    
    else:
        # === Single tensor grid mode ===
        imgs = images.detach().cpu()
        N = imgs.shape[0]
        if imgs.ndim == 3:
            # [N, H, W]
            to_plot = imgs.numpy()
            is_color = False
        elif imgs.ndim == 4 and imgs.shape[1] == 3:
            # [N, 3, H, W] -> [N, H, W, 3]
            to_plot = imgs.permute(0, 2, 3, 1).numpy()
            is_color = True
        else:
            raise ValueError(f"Unsupported shape {imgs.shape}, expected [N,H,W] or [N,3,H,W]")

        num_rows = math.ceil(N / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
        # axes = axes.flatten() if num_rows > 1 else [axes]
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < N:
                img = to_plot[i]
                if is_color:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap=cmap)
            ax.axis("off")

    # === Title & Save ===
    if title and str(title).strip():
        fig.suptitle(title, fontsize=font_size)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    plt.savefig(output_path, dpi=save_dpi)
    plt.close(fig)






def plot_lambda(lambd: list[torch.Tensor], timesteps: list[torch.Tensor], filename: str = "lambda_plot.png", save_dpi=300, show_legend=True):
    """
    Plots each k-value in the lambda tensor over time for every batch index.

    Args:
        lambd (list[Tensor]): list of T tensors, each of shape [BS, k]
        timesteps (list[Tensor]): list of T timesteps
        filename (str): output filename to save the plot
    """
    ts_values = [int(t.item()) for t in timesteps]  # [T]
    stacked = torch.stack(lambd, dim=0).float()     # [T, BS, k]
    T, BS, k = stacked.shape
    
    colors = plt.cm.get_cmap("tab10", k)
    
    fig, axs = plt.subplots(BS, 1, figsize=(12, 2 * BS), sharex=True)
    if BS == 1:
        axs = [axs]  # ensure iterable
    
    for b in range(BS):
        for ki in range(k):
            values = stacked[:, b, ki].cpu().numpy()  # [T]
            axs[b].plot(ts_values, values, label=f"k={ki}", color=colors(ki))
    
        axs[b].set_ylabel(f"Lambda for BS={b}")
        axs[b].grid(True)
        if show_legend:
            axs[b].legend(loc='upper right')
        axs[b].set_ylim(0, 1)
    
    axs[-1].set_xlabel("Timestep")
    plt.suptitle("Lambda values vs timesteps for each batch sample")
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, bbox_inches='tight', dpi=save_dpi)
    plt.close()






def plot_latent_grid(intermediates: dict, timesteps: list[torch.tensor], step: int = 1, filename: str = "latent_grid.png", batch_id: int = 0, projection=True, coupling=False, save_dpi=300):
    """
    Plot a 4-row grid of images:
    Row 1: mean_collapse latents
    Row 2: decoded latents
    Row 3: mean_collapse coupled_latents
    Row 4: decoded coupled latents
    Row 5: mean_collapse projected latents
    Row 6: decoded projected latents

    Args:
        intermediates: dict containing the 6 intermediate keys
        step: stride between timesteps to sample
        filename: file path to save the resulting plot
    """
    ts_values = [int(t.item()) for t in timesteps]
    
    B = batch_id
    T = len(intermediates["latents"])
    
    if ts_values[T-1] == 1:
        add_on = [] # skip duplicate t=1
    else:
        add_on = [T-1]
    
    selected_steps = list(range(0, T, step)) + add_on
    num_cols = len(selected_steps)
    
    num_rows = 2
    if projection:
        num_rows += 2
    if coupling:
        num_rows += 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, 10))

    for col_idx, t in enumerate(selected_steps):
        axes_idx = 0
        
        # Row 1: mean over channels of latent
        lat = intermediates["latents"][t][B]  # [lc, lh, lw]
        lat_mean = lat.mean(dim=0).cpu().numpy()
        axes[axes_idx, col_idx].imshow(lat_mean, cmap='viridis')
        axes[axes_idx, col_idx].set_title(f"t={ts_values[t]}")
        axes_idx += 1
        
        # Row 2: RGB decoded latent
        img1 = intermediates["latents_decoded"][t][B].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[axes_idx, col_idx].imshow(img1)
        axes_idx += 1
        
        if coupling:
            # Row 3: mean over channels of coupled latent
            lat_coup = intermediates["latents_coupled"][t][B]  # [lc, lh, lw]
            lat_coup_mean = lat_coup.mean(dim=0).cpu().numpy()
            axes[axes_idx, col_idx].imshow(lat_coup_mean, cmap='viridis')
            axes_idx += 1
            
            # Row 4: RGB decoded coupled latent
            img2 = intermediates["latents_coupled_decoded"][t][B].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[axes_idx, col_idx].imshow(img2)
            axes_idx += 1
        
        if projection:
            # Row 5: mean over channels of projected latent
            lat_proj = intermediates["latents_projected"][t][B]  # [lc, lh, lw]
            lat_proj_mean = lat_proj.mean(dim=0).cpu().numpy()
            axes[axes_idx, col_idx].imshow(lat_proj_mean, cmap='viridis')
            axes_idx += 1
            
            # Row 6: RGB decoded projected latent
            img2 = intermediates["latents_projected_decoded"][t][B].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[axes_idx, col_idx].imshow(img2)
            axes_idx += 1

    # Remove axis ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # Row labels
    row_labels = [
        "Latents\n(mean)",
        "Latents\nDecoded",
    ]
    if coupling:
        row_labels += [
            "Coupled\n(mean)",
            "Coupled\nDecoded"
        ]
    if projection:
        row_labels += [
            "Projected\n(mean)",
            "Projected\nDecoded"
        ]

    for row_idx, label in enumerate(row_labels):
        fig.text(0.01, 1 - (row_idx + 0.5) / num_rows, label, va='center', ha='left', fontsize=10)


    # plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=save_dpi)
    plt.close()





def plot_interpolated_images(decoded_images, interp_scales, filename="interpolated_images.png", save_dpi=300):
    num_images = decoded_images.size(0)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    
    for i in range(num_images):
        image = decoded_images[i].cpu()  # [C, H, W]
        image = TF.to_pil_image(image)
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(f"α={interp_scales[i].item():.1f}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=save_dpi)
    plt.close()




def plot_kde_l2_norms(named_tensors: dict[str, list[torch.Tensor]], filename: str = "kde_l2_norms.png", log_scale: bool = False):
    """
    Plots KDE of L2 norms for each key in the dictionary.

    Args:
        named_tensors (dict): keys -> list of tensors [BS, lc, lh, lw]
        filename (str): path to save the KDE plot
    """
    all_norms = {}
    for key, tensor_list in named_tensors.items():
        norms = []
        for tensor in tensor_list:
            flat = tensor.view(tensor.size(0), -1)      # [BS, D]
            l2 = torch.norm(flat, dim=1)                # [BS]
            norms += [float(i) for i in l2.cpu().numpy()]
        all_norms[key] = norms

    plt.figure(figsize=(10, 5))
    for key, values in all_norms.items():
        sns.kdeplot(values, label=key, fill=True)

    plt.xlabel("L2 Norm")
    
    if log_scale:
        plt.ylabel("Density (log scale)")
        plt.yscale("log")
    else:
        plt.ylabel("Density")
    plt.title("KDE Plot of L2 Norms for Each Key")
    plt.legend()
    plt.grid(True, which='both', axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def plot_hist_l2_norms(named_tensors: dict[str, list[torch.Tensor]], filename: str = "hist_l2_norms.png", log_scale: bool = False, bins=50):
    """
    Plots histograms of L2 norms for each key in the dictionary, with and without log scale.

    Args:
        named_tensors (dict): keys -> list of tensors [BS, lc, lh, lw]
        filename_prefix (str): base filename to save both linear and log histograms
    """
    all_norms = {}
    for key, tensor_list in named_tensors.items():
        norms = []
        for tensor in tensor_list:
            flat = tensor.view(tensor.size(0), -1)      # [BS, D]
            l2 = torch.norm(flat, dim=1)                # [BS]
            norms += [float(i) for i in l2.cpu().numpy()]
        all_norms[key] = norms
    
    # Flatten all values to determine shared bin range
    all_values = [v for group in all_norms.values() for v in group]
    bin_range = (min(all_values), max(all_values))
    bin_edges = np.linspace(*bin_range, num=bins + 1)  # +1 for edges

    # Log scale histogram
    if log_scale:
        plt.figure(figsize=(10, 5))
        for key, values in all_norms.items():
            plt.hist(values, bins=bin_edges, alpha=0.5, label=key, density=True)
        plt.xlabel("L2 Norm")
        plt.ylabel("Density (log scale)")
        plt.yscale("log")
        plt.title("Histogram of L2 Norms (Log Scale)")
        plt.legend()
        plt.grid(True, which='both', axis='y')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    # Linear scale histogram
    else:
        plt.figure(figsize=(10, 5))
        for key, values in all_norms.items():
            plt.hist(values, bins=bin_edges, alpha=0.5, label=key, density=True)
        plt.xlabel("L2 Norm")
        plt.ylabel("Density")
        plt.title("Histogram of L2 Norms (Linear Scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()



def split_intermediates_into_quartiles(kde_dict, intermediates_list, title="latents_projected"):
    n = len(intermediates_list)
    if n >= 4:
        q = n // 4
        kde_dict[f"{title}_q1"] = intermediates_list[:q]
        kde_dict[f"{title}_q2"] = intermediates_list[q:2*q]
        kde_dict[f"{title}_q3"] = intermediates_list[2*q:3*q]
        kde_dict[f"{title}_q4"] = intermediates_list[3*q:]
    elif n == 3:
        kde_dict[f"{title}_q1"] = [intermediates_list[0]]
        kde_dict[f"{title}_q2"] = [intermediates_list[1]]
        kde_dict[f"{title}_q3"] = [intermediates_list[2]]
    elif n == 2:
        kde_dict[f"{title}_q1"] = [intermediates_list[0]]
        kde_dict[f"{title}_q2"] = [intermediates_list[1]]
    elif n == 1:
        kde_dict[f"{title}_q1"] = intermediates_list
    return kde_dict
