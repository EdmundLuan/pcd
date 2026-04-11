import os
import sys
import logging
import datetime
import click
import pdb
import numpy as np
import importlib.util

from tqdm import tqdm
from itertools import product
from os.path import join, dirname

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR    = dirname(os.path.abspath(__file__))
PROJECT_ROOT= dirname(dirname(BASE_DIR))
CONFIG_DIR = join(PROJECT_ROOT, "configs")
LOG_DIR     = os.path.join(PROJECT_ROOT, "results/000event_logs")
GUIDERS     = ['vanilla', 'coupling', 'coupling_ps']
PROJECTORS  = ['none', 'max_vel_admm']

PROJECTOR_PARAMS = {
    'max_vel_admm': {
        '_target_': 'ADMMProjectionOperatorCUDAGraph', 
        'rho': 10.0,
        'max_iter': 1000,
        'tol': 3.0e-6,
        'decomp': 'lu',
        'convergence_check': True,
        'check_period': 100,
        'verbose': False, 
    }
}


# Active sweep defaults
n_init_states   = 51  
batch_size      = 32
num_agents      = [2, 3, 4, 6, 8]  
default_cost_fn_key = "hinge_sqr_l2"
projector       = 'max_vel_admm'
vmaxs = [0.781]
scales = [1e-1]
skips  = []

# Helpers
def import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    logger.info(f"Imported module '{module_name}' from '{file_path}'")
    return module

# CLI
@click.command(context_settings={"ignore_unknown_options": True,
                                 "allow_extra_args": True})
@click.option("--config", type=str, required=True,
              help="Path to the configuration YAML file.")
@click.option("--work-dir",    type=str, default=BASE_DIR)
@click.option("--gpu-id",      type=int, default=0)
@click.option("--cost-func-key","-c",  type=str, default=default_cost_fn_key)
@click.option("--product-params","-r", is_flag=True)
@click.option("--comments", "-m",        type=str, default="")
@click.pass_context
def main(ctx, config, work_dir, gpu_id, cost_func_key, product_params, comments):
    # grab any --foo=bar or foo=bar tokens
    raw_overrides = list(ctx.args)

    # sanity
    assert os.path.isdir(LOG_DIR), f"LOG_DIR not found: {LOG_DIR}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    ### Now import GPU-specific modules
    from mmd.common.argparse import load_config, parse_overrides

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_nm = f"eval_runs-{timestamp}" + (f"-{comments}" if comments else "") + ".log"
    log_pth = os.path.join(LOG_DIR, log_nm)
    logger.info(f"Logging all runs to {log_pth}")
    if raw_overrides:
        logger.info(f"Ad-hoc overrides: {raw_overrides}")

    # import sub-scripts
    guide_py    = os.path.join(work_dir, "inference_pcdiff.py")
    base_py     = os.path.join(work_dir, "inference_pcdiff.py")
    eval_guide  = import_module_from_path("eval_guide", guide_py)
    eval_base   = import_module_from_path("eval_guide", base_py)

    cfg_dir = CONFIG_DIR

    grid = (product(num_agents, scales, vmaxs) if product_params
            else zip(num_agents, scales, vmaxs))
    total = (len(num_agents)*len(scales)*len(vmaxs)
                if product_params else len(num_agents))

    for trial, (n_agent, scl, v_max) in enumerate(tqdm(grid, total=total, desc="PCD runs", ncols=80)):
        logger.info(f"Running coupling+proj {cost_func_key} n_agent={n_agent}, scl={scl}, v_max={v_max}…")
        args_list = [
            f"--system.log_pth={log_pth!r}",
            f"--experiment.num_agents={n_agent}",
            f"--experiment.n_inits={n_init_states}",
            f"--planner.projection.project_params.vel_max={v_max}",
            f"--planner.single_agent.planner_alg='pcd'",
            f"--planner.single_agent.cost_func_key={cost_func_key}",
            f"--planner.single_agent.n_guide_steps=1",
            f"--planner.single_agent.weight_grad_cost_constraints={scl}",
            f"--planner.single_agent.n_samples={batch_size}",
            f"--planner.single_agent.timeit=True",
        ]
        for k, v in PROJECTOR_PARAMS[projector].items():
            args_list.append(f"--planner.projection.projector.{k}={v}")
        args_list += raw_overrides

        overrides = parse_overrides(args_list)
        cfg = load_config(join(cfg_dir, config), overrides)
        comments_this_run = (f"timing_pcd"
                    + f"-{cost_func_key}-N{n_agent}-B{batch_size}-scl{scl}-v{v_max}"
                    + (f"-{comments}" if comments else ""))
        eval_guide.run_planning_experiment(cfg, comments=comments_this_run)
        logger.info(f"Done n_agent={n_agent}, scl={scl}, v_max={v_max}.")


if __name__ == "__main__":
    main()
