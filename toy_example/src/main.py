"""Main driver script to run various LMCs."""

import argparse
from copy import deepcopy
from src.lmc import run_main
from src.lmc import (
    run_conditional_lmc,
    run_projected_coupled_lmc,
    run_projected_lmc,
    run_coupled_lmc
)


# python -m src.main


# BASE_CONFIG
# =============================================================
# =============================================================

BASE_CONFIG = argparse.Namespace()

# random seed
BASE_CONFIG.seed       = 3407

# Langevin steps
BASE_CONFIG.N          = 500000          # batch size (pairs)
BASE_CONFIG.steps      = 200             # Langevin steps
BASE_CONFIG.eta        = 0.05            # step size

# init dist & score function
BASE_CONFIG.mu         = 4.5             # Guassian mean (score func)
BASE_CONFIG.sigma2_x   = 0.6             # Guassian var (score func)
BASE_CONFIG.sigma2_y   = 1.0             # Guassian var (score func)

# sphere / rectangle sizes
BASE_CONFIG.rx         = 3.0             # each sphere's radius size
BASE_CONFIG.ry         = 1.0

# coupling
BASE_CONFIG.d0         = BASE_CONFIG.rx + BASE_CONFIG.ry   # desired separation = 4
BASE_CONFIG.lam        = 32                                # coupling strength λ
BASE_CONFIG.sqrt_2eta  = (2.0 * BASE_CONFIG.eta) ** 0.5    # 
BASE_CONFIG.alpha      = 1.5                               # Log-repulsive c(x,y) +alpha

# projection
BASE_CONFIG.corridor_min = 0  # corridor min
BASE_CONFIG.corridor_max = 9  # corridor max

# visualisation
BASE_CONFIG.cmap_non_collide = "Blues"
BASE_CONFIG.cmap_collide     = "Oranges"
BASE_CONFIG.view_min   = -0.25
BASE_CONFIG.view_max   = 9.25

# save
BASE_CONFIG.output_dir = "plots/lmc"

# =============================================================
# =============================================================



if __name__ == "__main__":
    
    # custom config
    COUPLED_CFG           = deepcopy(BASE_CONFIG)
    PROJECTED_CFG         = deepcopy(BASE_CONFIG)
    PROJECTED_COUPLED_CFG = deepcopy(BASE_CONFIG)
    CONDITIONAL_LMC_CFG   = deepcopy(BASE_CONFIG)
    CONDITIONAL_LMC_CFG.lam = 12
    
    
    
    FN_MAP = {
        "coupled_lmc"           : run_coupled_lmc,
        "projected_coupled_lmc" : run_projected_coupled_lmc,
        "conditional_lmc"       : run_conditional_lmc,
        # "projected_lmc"         : run_projected_lmc,
    }
    
    CONFIG_MAP = {
        "coupled_lmc"           : COUPLED_CFG,
        "projected_coupled_lmc" : PROJECTED_COUPLED_CFG,
        "conditional_lmc"       : CONDITIONAL_LMC_CFG,
        # "projected_lmc"         : PROJECTED_CFG,
    }
    run_main(CONFIG_MAP, FN_MAP)
