"""Pipeline script for each LMC variation."""

import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle




# Operators & Gradients
# ======================================
def proj(z, lo, hi):
    """Simple projection via clamp."""
    return torch.clamp(z, min=lo, max=hi)

def dC_dx(x, y, alpha):
    """C(x,y): Log-Repulsion, Eq. (10) in paper."""
    eps   = 1e-12
    delta = x - y
    r     = torch.sqrt(delta * delta + eps)
    return - delta / (r * (r + alpha))

def dC_dy(x, y, alpha):
    return -dC_dx(x, y, alpha)
# ======================================




def run_coupled_lmc(seed, N, steps, mu_x, mu_y, sigma2_x, sigma2_y, eta, lam, sqrt_2eta, alpha, x_lo, x_hi, y_lo, y_hi, device):
    """Coupling Only."""
    torch.manual_seed(seed)
    
    # ---- init: sample centers from N(mu, I)
    X = torch.normal(mean=mu_x, std=sigma2_x**0.5, size=(N,), device=device)
    Y = torch.normal(mean=mu_y, std=sigma2_y**0.5, size=(N,), device=device)
    
    # ---- Langevin loop
    for _ in tqdm(range(steps), desc="Coupled LMC"):
        # Gaussian prior score terms
        drift_x = (mu_x - X) / sigma2_x
        drift_y = (mu_y - Y) / sigma2_y
        
        # coupling (penalty) gradients
        g_x = dC_dx(X, Y, alpha)
        g_y = dC_dy(X, Y, alpha)
        
        # Euler–Maruyama step
        X = X + eta * (drift_x - lam * g_x) + sqrt_2eta * torch.randn_like(X)
        Y = Y + eta * (drift_y - lam * g_y) + sqrt_2eta * torch.randn_like(Y)
    
    # ---- to numpy
    Xc = X.detach().cpu().numpy()
    Yc = Y.detach().cpu().numpy()
    
    return Xc, Yc



def run_projected_lmc(seed, N, steps, mu_x, mu_y, sigma2_x, sigma2_y, eta, lam, sqrt_2eta, alpha, x_lo, x_hi, y_lo, y_hi, device):
    """Projection Only."""
    torch.manual_seed(seed)
    
    # ---- init: sample centers from N(mu, I)
    X = torch.normal(mean=mu_x, std=sigma2_x**0.5, size=(N,), device=device)
    Y = torch.normal(mean=mu_y, std=sigma2_y**0.5, size=(N,), device=device)
    
    # ---- Langevin loop
    for _ in tqdm(range(steps), desc="Projected LMC"):
        # Gaussian prior score terms
        drift_x = (mu_x - X) / sigma2_x
        drift_y = (mu_y - Y) / sigma2_y
        
        # Euler–Maruyama step + proj
        X = proj(X + eta * drift_x + sqrt_2eta * torch.randn_like(X), x_lo, x_hi)
        Y = proj(Y + eta * drift_y + sqrt_2eta * torch.randn_like(Y), y_lo, y_hi)
        
    # ---- to numpy
    Xc = X.detach().cpu().numpy()
    Yc = Y.detach().cpu().numpy()
    
    return Xc, Yc



def run_projected_coupled_lmc(seed, N, steps, mu_x, mu_y, sigma2_x, sigma2_y, eta, lam, sqrt_2eta, alpha, x_lo, x_hi, y_lo, y_hi, device):
    """Projection + Coupling."""
    torch.manual_seed(seed)
    
    # ---- init: sample centers from N(mu, I)
    X = torch.normal(mean=mu_x, std=sigma2_x**0.5, size=(N,), device=device)
    Y = torch.normal(mean=mu_y, std=sigma2_y**0.5, size=(N,), device=device)
    
    # ---- Langevin loop
    for _ in tqdm(range(steps), desc="Projected Coupled LMC"):
        # Gaussian prior score terms
        drift_x = (mu_x - X) / sigma2_x
        drift_y = (mu_y - Y) / sigma2_y
        
        # coupling (penalty) gradients
        g_x = dC_dx(X, Y, alpha)
        g_y = dC_dy(X, Y, alpha)
        
        # Euler–Maruyama step + projection
        X = proj(X + eta*(drift_x - lam*g_x) + sqrt_2eta*torch.randn_like(X), x_lo, x_hi)
        Y = proj(Y + eta*(drift_y - lam*g_y) + sqrt_2eta*torch.randn_like(Y), y_lo, y_hi)
    
    # ---- to numpy
    Xc = X.detach().cpu().numpy()
    Yc = Y.detach().cpu().numpy()
    
    return Xc, Yc



def run_conditional_lmc(seed, N, steps, mu_x, mu_y, sigma2_x, sigma2_y, eta, lam, sqrt_2eta, alpha, x_lo, x_hi, y_lo, y_hi, device):
    """Conditional Sampling (Sequential)."""
    torch.manual_seed(seed)
    
    # sample x and fix it
    X = torch.normal(mean=mu_x, std=sigma2_x**0.5, size=(N,), device=device)
    if steps > 0:
        for _ in tqdm(range(steps), desc="Regular LMC"):
            score_x = (mu_x - X) / sigma2_x
            X = X + eta * score_x + sqrt_2eta * torch.randn_like(X)
    X_star = X.detach()
    
    # sample y w.r.t. fixed x / X_star
    Y = torch.normal(mean=mu_y, std=sigma2_y**0.5, size=(N,), device=device)
    for _ in tqdm(range(steps), desc="Conditional LMC"):
        score_prior_y = (mu_y - Y) / sigma2_y
        g_y = dC_dy(X_star, Y, alpha)
        Y = Y + eta * (score_prior_y - lam * g_y) + sqrt_2eta * torch.randn_like(Y)
    
    # ---- to numpy
    Xc = X_star.detach().cpu().numpy()
    Yc = Y.detach().cpu().numpy()
    
    return Xc, Yc






def run_main(CONFIG_MAP, FN_MAP):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # histogram num bins
    bins = 180
    
    # fontsize
    fontsize = 20
    
    print("\nRunning various LMCs")
    print("============================")
    
    BUCKET = {}
    
    for MODE, LMC_FN in FN_MAP.items():
        
        print(f"Running {MODE}..")
        
        # read config
        # ==================================
        CONFIG = CONFIG_MAP[MODE]
        
        N            = CONFIG.N
        steps        = CONFIG.steps
        eta          = CONFIG.eta
        mu           = CONFIG.mu
        sigma2_x     = CONFIG.sigma2_x
        sigma2_y     = CONFIG.sigma2_y
        rx           = CONFIG.rx
        ry           = CONFIG.ry
        d0           = CONFIG.d0
        lam          = CONFIG.lam
        sqrt_2eta    = CONFIG.sqrt_2eta
        alpha        = CONFIG.alpha
        corridor_min = CONFIG.corridor_min
        corridor_max = CONFIG.corridor_max
        view_min     = CONFIG.view_min
        view_max     = CONFIG.view_max
        # ==================================
        
        
        # feasible intervals (hard walls via projection)
        x_lo, x_hi = corridor_min+rx, corridor_max-rx
        y_lo, y_hi = corridor_min+ry, corridor_max-ry
        
        # feasible rectangle
        rect_x, rect_w = float(x_lo), float(x_hi - x_lo)
        rect_y, rect_h = float(y_lo), float(y_hi - y_lo)
        
        
        # run each LMC variant
        x, y = LMC_FN(
            seed      = CONFIG.seed,
            N         = N,
            steps     = steps,
            mu_x      = mu,
            mu_y      = mu,
            sigma2_x  = sigma2_x,
            sigma2_y  = sigma2_y,
            eta       = eta,
            lam       = lam,
            sqrt_2eta = sqrt_2eta,
            alpha     = alpha,
            x_lo      = x_lo,
            x_hi      = x_hi,
            y_lo      = y_lo,
            y_hi      = y_hi,
            device    = device
        )
        
        # keep only non-colliding samples & allow anything outside the wall
        r_xy   = np.abs(x - y)
        non_collide_mask   = (r_xy >= d0)
        collide_mask = ~non_collide_mask
        x_non_collide = x[non_collide_mask]
        y_non_collide = y[non_collide_mask]
        x_collide = x[collide_mask]
        y_collide = y[collide_mask]
        
        # histogram count
        counts, xedges, yedges = np.histogram2d(
            x_non_collide, y_non_collide, bins=bins, 
            range=[[view_min, view_max], [view_min, view_max]]
        )
        coll_counts, xedges_c, yedges_c = np.histogram2d(
            x_collide, y_collide, bins=bins, 
            range=[[view_min, view_max], [view_min, view_max]]
        )
        
        # norm to probs
        total = N
        probs = counts / (total if total > 0 else 1)
        
        # natural log
        log_probs = np.full_like(probs, 0.0, dtype=float)
        mask = probs > 0
        log_probs[mask] = np.log(probs[mask])
        
        log_counts = np.full_like(counts, 0.0, dtype=float)
        mask = counts > 0
        log_counts[mask] = np.log(counts[mask])
        log_counts_c = np.full_like(coll_counts, 0.0, dtype=float)
        mask_c = coll_counts > 0
        log_counts_c[mask_c] = np.log(coll_counts[mask_c])
        
        # proportion of valid samples (inside wall & do not collide)
        # 1) Inside feasible intervals (both x and y inside their walls)
        inside_mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        num_inside   = int(inside_mask.sum())
        prop_inside  = num_inside / (N if N > 0 else 1)
        
        # 2) Do not collide (distance >= d0), regardless of walls
        non_collide_mask = (np.abs(x - y) >= d0)
        num_non_collide  = int(non_collide_mask.sum())
        prop_non_collide = num_non_collide / (N if N > 0 else 1)
        
        # 3) Both conditions simultaneously
        both_mask        = inside_mask & non_collide_mask
        num_both         = int(both_mask.sum())
        prop_both        = num_both / (N if N > 0 else 1)
        
        # store
        BUCKET[MODE] = {
            "x"         : x,
            "y"         : y,
            "counts"    : counts,
            "xedges"    : xedges,
            "yedges"    : yedges,
            "coll_counts": coll_counts,
            "xedges_c"  : xedges_c,
            "yedges_c"  : yedges_c,
            "total"     : total,
            "probs"     : probs,
            "log_probs" : log_probs,
            "log_counts": log_counts,
            "log_counts_c": log_counts_c,
            "prop_inside"     : prop_inside,
            "prop_non_collide": prop_non_collide,
            "prop_valid"      : prop_both
        }
    
    print("============================")
    
    
    
    # Set colorbar to cover from 0-99 th percentile values.
    logs_all = []
    for d in BUCKET.values():
        c = d["counts"]
        v = np.log(c[c > 0].ravel())
        if v.size:
            logs_all.append(v)
    
    if logs_all:
        p90_log = np.percentile(np.concatenate(logs_all), 99)
    else:
        p90_log = 0.0  # fallback if everything is empty
    
    print(f"Global 90th percentile (log-counts): {p90_log:.6g}")
    
    colorbar_norm = colors.Normalize(vmin=0.0, vmax=p90_log)
    
    
    
    print(f"\nPlotting Heatmaps")
    print("============================")
    
    
    for MODE, RESULTS in BUCKET.items():
        
        print(f"Plotting {MODE}..")
        
        # read config
        # ==================================
        CONFIG = CONFIG_MAP[MODE]
        rx               = CONFIG.rx
        ry               = CONFIG.ry
        corridor_min     = CONFIG.corridor_min
        corridor_max     = CONFIG.corridor_max
        x_lo, x_hi       = corridor_min+rx, corridor_max-rx
        y_lo, y_hi       = corridor_min+ry, corridor_max-ry
        rect_x, rect_w   = float(x_lo), float(x_hi - x_lo)
        rect_y, rect_h   = float(y_lo), float(y_hi - y_lo)
        cmap_non_collide = cm.get_cmap(CONFIG.cmap_non_collide)
        cmap_collide     = cm.get_cmap(CONFIG.cmap_collide)
        view_min         = CONFIG.view_min
        view_max         = CONFIG.view_max
        output_dir       = CONFIG.output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # ==================================
        
        
        
        # segment out density map regions that are 0.
        # =================================================
        log_nc = RESULTS["log_counts"]
        counts = RESULTS["counts"]
        log_nc = np.ma.masked_where(counts == 0, log_nc)
        
        log_c = RESULTS["log_counts_c"]
        coll_counts = RESULTS["coll_counts"]
        log_c = np.ma.masked_where(coll_counts == 0, log_c)
        
        cmap_nc = deepcopy(cmap_non_collide)
        cmap_nc.set_bad((0, 0, 0, 0))   # RGBA, alpha=0
        
        cmap_c  = deepcopy(cmap_collide)
        cmap_c.set_bad((0, 0, 0, 0))
        # =================================================
        
        
        
        # build marginal histograms
        # =================================================
        x = RESULTS["x"]
        y = RESULTS["y"]
        
        grid_min = min(x.min(), y.min(), x_lo)
        grid_max = max(x.max(), y.max(), x_hi)
        bins     = min(200, max(10, min(len(x),len(y))//5))
        edges    = np.linspace(grid_min, grid_max, bins+1)
        centers  = 0.5*(edges[:-1]+edges[1:])
        widths   = edges[1:] - edges[:-1]
        hx,_     = np.histogram(x, bins=edges)
        hy,_     = np.histogram(y, bins=edges)
        pdf_x    = hx/(len(x)*widths);  pdf_x /= np.trapz(pdf_x, centers)
        pdf_y    = hy/(len(y)*widths);  pdf_y /= np.trapz(pdf_y, centers)
        # =================================================
        
        
        
        # construct layout with GridSpec
        fig = plt.figure(figsize=(8,8))
        gs  = gridspec.GridSpec(2,2,
                width_ratios =[4,1],
                height_ratios=[1,4],
                wspace=0.05, hspace=0.05)
        
        ax_histy = fig.add_subplot(gs[0,0])
        ax_joint = fig.add_subplot(gs[1,0])
        ax_histx = fig.add_subplot(gs[1,1])
        
        
        # 1) top histogram of x
        ax_histx.fill_betweenx(centers, 0, pdf_x, color='purple', alpha=0.3, step='mid', linewidth=0)
        ax_histx.plot(pdf_x, centers, '-', color='purple', markersize=4, linewidth=2)
        ax_histx.axhline(corridor_min, color='black', linestyle='-', alpha=0.75, linewidth=3)
        ax_histx.axhline(corridor_max, color='black', linestyle='-', alpha=0.75, linewidth=3)
        ax_histx.axhline(x_lo, color='red', linestyle='--', alpha=1.0, linewidth=3)
        ax_histx.axhline(x_hi, color='red', linestyle='--', alpha=1.0, linewidth=3)
        ax_histx.set_ylim(view_min, view_max)
        ax_histx.set_xlim(0, 1)
        ax_histx.set_xticks([])
        ax_histx.tick_params(labelleft=False)
        
        # 2) right histogram of y
        ax_histy.fill_between(centers, pdf_y, 0, color='green', alpha=0.3, step='mid', linewidth=0)
        ax_histy.plot(centers, pdf_y, '-', color='green', markersize=4, linewidth=2)
        ax_histy.axvline(corridor_min, color='black', linestyle='-', alpha=0.75, linewidth=3)
        ax_histy.axvline(corridor_max, color='black', linestyle='-', alpha=0.75, linewidth=3)
        ax_histy.axvline(y_lo, color='red', linestyle='--', alpha=1.0, linewidth=3)
        ax_histy.axvline(y_hi, color='red', linestyle='--', alpha=1.0, linewidth=3)
        ax_histy.set_xlim(view_min, view_max)
        ax_histy.set_ylim(0, 1)
        ax_histy.set_yticks([])
        ax_histy.tick_params(labelbottom=False)
        
        # show segmented density maps
        ax_joint.imshow(
            log_nc,
            origin='lower',
            extent=[view_min, view_max, view_min, view_max],
            interpolation='none',
            cmap=cmap_nc,
            norm=colorbar_norm,
            zorder=1,
        )
        ax_joint.imshow(
            log_c,
            origin='lower',
            extent=[view_min, view_max, view_min, view_max],
            interpolation='none',
            cmap=cmap_c,
            norm=colorbar_norm,
            zorder=2,
        )
        # draw feasible regions
        rect = Rectangle(
            (rect_y, rect_x), rect_h, rect_w,
            fill=False,
            edgecolor='red',
            linewidth=3,
            linestyle='--',
            alpha=0.8,
            zorder=3
        )
        ax_joint.add_patch(rect)
        rect = Rectangle(
            (corridor_min, corridor_min), (corridor_max-corridor_min), (corridor_max-corridor_min),
            fill=False,
            edgecolor='black',
            linewidth=3,
            alpha=0.75,
            zorder=4
        )
        ax_joint.add_patch(rect)
        
        # set lims and labels
        ax_joint.set_xlim(view_min, view_max)
        ax_joint.set_ylim(view_min, view_max)
        ax_joint.set_xlabel("Location of Small Block", fontsize=fontsize)
        ax_joint.set_ylabel("Location of Big Block", fontsize=fontsize)
        ax_joint.set_xticks(np.linspace(corridor_min, corridor_max, len(range(corridor_max - corridor_min + 1))))
        ax_joint.set_yticks(np.linspace(corridor_min, corridor_max, len(range(corridor_max - corridor_min + 1))))
        ax_joint.tick_params(axis='both', labelsize=fontsize)
        ax_joint.grid(color='black', alpha=0.1, linewidth=0.7)
        
        # clean up ticks
        for ax in (ax_joint,):
            ax.tick_params(pad=1, labelsize=12)
        for ax in (ax_histx, ax_histy):
            ax.margins(0)
        
        # tighter tick-label padding for paper
        ax_joint.margins(x=0, y=0)
        ax_joint.tick_params(pad=1)
        ax_joint.xaxis.labelpad = 1
        ax_joint.yaxis.labelpad = 1
        
        # add proportion numbers
        prop_valid = RESULTS["prop_valid"]
        prop_inside = RESULTS["prop_inside"]
        prop_non_collide = RESULTS["prop_non_collide"]
        
        label = (
            # f"$p_\\mathrm{{inside}}$ = {prop_inside:.1%}\n"
            f"$p_\\mathrm{{out}}$ = {1-prop_inside:.1%}\n"
            f"$p_\\mathrm{{overlap}}$ = {1-prop_non_collide:.1%}"
            # f"$p_\\mathrm{{feasible}}$ = {prop_valid:.1%}"
        )
        ax_joint.text(
            0.9, 0.9, label, transform=ax_joint.transAxes,
            ha='right', va='top', fontsize=fontsize, linespacing=1.2,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'),
            zorder=5
        )
        
        plt.savefig(os.path.join(output_dir, f'heatmap_nc_{MODE}_2.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    print("============================")
    print("DONE.")
