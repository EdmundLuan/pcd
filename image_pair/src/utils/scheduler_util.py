"""Implementation of Schedulers. All should return decreasing values, and in range [0,1]."""

import math
import torch



def linear_scheduler(T, **kwargs):
    """
    Generates a linear scheduler that decreases linearly from 1 to 0 over T timesteps.
    
    Args:
        T (int): The number of timesteps.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (T,) with values linearly spaced 
                      from 1.0 to 0.0.
    """
    linear = torch.linspace(1., 0., steps=T)
    return linear


def cosine_scheduler(T, s=0.0, **kwargs):
    """
    Generates a cosine-based alpha_bar schedule as described in DDPM 
    with optional offset `s`.
    
    Args:
        T (int): The number of timesteps.
        s (float, optional): Small offset added to the time index in cosine 
                             decay for smooth start (default: 0.0).
    
    Returns:
        torch.Tensor: A 1D tensor of shape (T,) representing ᾱₜ values 
                      that smoothly decrease from 1 to ~0.
    
    Value Range (with s=0.0):
        - Starts near 1.0 (exactly 1.0 when t=0)
        - Ends near 0.0 (but not exactly 0 due to floating point precision)
    """
    def alpha_bar(t):
        return math.cos((t / (T - 1) + s) / (1 + s) * math.pi / 2) ** 2
    
    alpha_bars = [alpha_bar(t) for t in range(T)]
    alpha_bars[-1] = 0.    # force exact 0 due to near-zero float from precision limits
    return torch.tensor(alpha_bars)


def convex_scheduler(T, r=2, **kwargs):
    """
    Generates a convex decay curve that decreases from 1 to 0,
    with a sharp drop early on and a slow tail.
    
    Args:
        T (int): The number of timesteps.
        r (float): Radius control. Larger r makes the arc flatter.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (T,) with values following a convex decay.
    """
    t = torch.linspace(0, 1, T)
    return 1 - t**(1/r)


def concave_scheduler(T, r=2, **kwargs):
    """
    Generates a concave decay curve that decreases from 1 to 0,
    with a gentle slope initially and a sharp drop near the end.
    
    Args:
        T (int): The number of timesteps.
        r (float): Radius control. Larger r makes the arc flatter.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (T,) with values following a concave decay.
    """
    t = torch.linspace(0, 1, T)
    return 1 - t**r


def concave_sphere_scheduler(T, r=2, **kwargs):
    """
    Generates a concave curve resembling the upper half of a circle (semicircle arch),
    decreasing from 1 to 0.
    
    Args:
        T (int): Number of timesteps.
        r (float): Radius control. Larger r makes the arc flatter.
    
    Returns:
        torch.Tensor: Concave schedule (1 to 0) over T steps.
    """
    t = torch.linspace(0, 1, T)
    return torch.sqrt(1 - t**r)


def convex_sphere_scheduler(T, r=2, **kwargs):
    """
    Generates a convex curve that is the full mirror (horizontal + vertical)
    of the concave_sphere_scheduler. Resembles the lower half of a circle, 
    decreasing from 1 to 0.
    
    Args:
        T (int): Number of timesteps.
        r (float): Radius control. Larger r makes the arc flatter.
    
    Returns:
        torch.Tensor: Convex schedule (1 to 0) over T steps.
    """
    t = torch.linspace(0, 1, T)
    return 1 - torch.sqrt(1 - (1 - t)**r)




SCHEDULER_MAPPING = {
    "linear_scheduler"         : linear_scheduler,
    "cosine_scheduler"         : cosine_scheduler,
    "convex_scheduler"         : convex_scheduler,
    "concave_scheduler"        : concave_scheduler,
    "convex_sphere_scheduler"  : convex_sphere_scheduler,
    "concave_sphere_scheduler" : concave_sphere_scheduler
}



if __name__ == "__main__":
    
    # Run via: python -m src.utils.scheduler_util
    
    import matplotlib.pyplot as plt
    
    # Parameters
    T = 500
    R = 2
    linear  = linear_scheduler(T, r=R)
    cosine  = cosine_scheduler(T, r=R)
    convex  = convex_scheduler(T, r=R)
    concave = concave_scheduler(T, r=R)
    convex_sphere = convex_sphere_scheduler(T, r=R)
    concave_sphere = concave_sphere_scheduler(T, r=R)
    
    assert len(linear) == T
    assert len(cosine) == T
    assert len(convex) == T
    assert len(concave) == T
    assert len(convex_sphere) == T
    assert len(concave_sphere) == T
    
    print(f"linear : {linear[0]}, {linear[-1]}")
    print(f"cosine : {cosine[0]}, {cosine[-1]}")
    print(f"convex : {convex[0]}, {convex[-1]}")
    print(f"concave: {concave[0]}, {concave[-1]}")
    print(f"convex_sphere : {convex_sphere[0]}, {convex_sphere[-1]}")
    print(f"concave_sphere: {concave_sphere[0]}, {concave_sphere[-1]}")
    
    # Plotting
    Y = 1
    plt.figure(figsize=(10, Y * 10))
    plt.plot(list(reversed(range(T))), Y * linear, label='linear', color='#1f77b4')                 # muted blue
    plt.plot(list(reversed(range(T))), Y * cosine, label='cosine', color='#ff7f0e')                 # vivid orange
    plt.plot(list(reversed(range(T))), Y * convex, label='convex', color='#2ca02c')                 # fresh green
    plt.plot(list(reversed(range(T))), Y * concave, label='concave', color='#17becf')               # cyan-teal
    plt.plot(list(reversed(range(T))), Y * convex_sphere, label='convex_sphere', color='#e377c2')   # pink
    plt.plot(list(reversed(range(T))), Y * concave_sphere, label='concave_sphere', color='#8c564b') # dark brown
    plt.xlabel("Timestep t")
    plt.ylabel("Value")
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.title("Schedulers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/utils/schedulers.png")
