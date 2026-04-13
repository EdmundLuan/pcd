# Diffusion-Policy-based PCD Implementation

**Table of Contents**
- [Diffusion-Policy-based PCD Implementation](#ltldog-diffusion-policy-based-implementation)
    - [Installation](#installation)
    - [Play with the Env (Optional)](#play-with-the-env-optional)
    - [Pretrained Weights](#pretrained-weights)
    - [Inference](#inference-with-pcd)
    - [Result Analysis](#result-analysis)
    - [References](#references)


This implementaion builds upon [<span style="font-variant: small-caps;">Diffusion Policy</span>](https://github.com/real-stanford/diffusion_policy)[1] and [<span style="font-variant: small-caps;">LTLDoG</span>](https://github.com/clear-nus/ltldog)[2]. 


## Installation

1. Create a Conda environment with `conda_environment.yaml` (we also recommend Mambaforge) 

    ```sh
    mamba env create -f conda_environment.yaml
    # Or just use conda
    # conda env create -f conda_environment.yaml
    ```

    Downgrade `huggingface-hub` for compatibility

    ```sh
    pip install "huggingface_hub<0.14.0"
    ```

2. Install `diffusion_policy` as a package in the newly created environment

    ```sh
    mamba activate pcdp 
    pip install -e . 
    ```


## Play with the PushT Task (Optional)

The demo script inherits from <span style="font-variant: small-caps;">Diffusion Policy</span>. 
Familiarize yourself with the environment by 

```sh
python demo_pusht.py --help
```

## Pretrained Model Weights

- Pretained weights are from <span style="font-variant: small-caps;">LTLDoG</span> [1]; check its [official code release](https://github.com/clear-nus/ltldog) to download the weights for the PushT task. 
- Move the checkpoint(s) to `data/pretrained/diffusion` for inference. 
- See [Inference Configs](#inference-configs) for configuring the weights path. 

### Configs

The configuration setting pipeline inherits from <span style="font-variant: small-caps;">Diffusion Policy</span> [1] and <span style="font-variant: small-caps;">LTLDoG</span> [2]. 


## Inference with PCD

Inference scripts are under `scripts/`:
- We provide a sequential executing scripts that could be used to run multiple experiments (for parameter sweeps). 
- Run `eval_H16_pusht_seq.py` with proper args for inference. ***Set up hyperparameters in the script first***. 
- Calling example: 
    ```bash
    python scripts/eval_H16_seq.py  -r  --gpu-id 0  --guider coupling  --cost-func-key "dpp"  --projector max_vel_admm  -m "custom_comments"
    ```
    - Args: 
        - `--guider` denotes the type of coupling costs and should be one of `{'vanilla', 'coupling', 'coupling_ps'}`, corresponding to no coupling at all, standard coupling costs, or posteior sampling cost variants. 
        - `--cost-func-key` should be in `{'dpp', 'sum_log_l2'}`. 
        - `--projector` should be one of `{'none', 'max_vel_admm'}`. 
        - `-r` means running with the *Cartesian product* of all hyperparameters set in the script. Check the details in the script. 
- Results are recorded by default under `logs/tests/`. 
- A log will be saved under `logs/tests/event_logs`, containing result directories of (potentially multiple) experiments. 


### Inference Configs

- Configuration files are at `diffusion_policy/config/eval_*.yaml`. You may need to change the `diffusion_checkpoint` field. 


## Result Analysis
1. Edit the config file for analysis `config/analysis_config.yaml`: 
    - Paste the relative directory to the log file `xxx.log` under the `log_paths` field. 
    - Configure other switches and default constants if needed: 
        - E.g., `DEFAULT_MAX_VEL` specifies the value of max-velocity constraint; *this only affects methods that are NOT configured with such info*, e.g., vanilla diffusion or coupling-only method. The analysis script automatically retrieves the velocity constraint used for PCD or projection-only method. 
2. Run the analysis script `scripts/analyzing_results.py` to calculate all the metrics: 
    ```bash
    python scripts/analyzing_results.py --config config/analysis_config.yaml 
    ```
    - A JSON file will be generated and placed under each directory specified in the log file; this JSON file contains all the metrics. 
    - You need to run the analysis script multiple times if you are calculating velocity constraint satisfaction rates for *vanilla diffusion or coupling-only* method *for different velocity thresholds*, as those are velocity-constraint-agnostic.  

---
## ✏️ Citation 
If you find this repo or the ideas presented in our paper useful for your research, please consider citing our paper.
```
@inproceedings{luan2026projected,
    title={Projected Coupled Diffusion for Test-Time Constrained Joint Generation},
    author={Hao Luan and Yi Xian Goh and See-Kiong Ng and Chun Kai Ling},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=1FEm5JLpvg}
}
```

---
## References

- [1] Chi C, Xu Z, Feng S, Cousineau E, Du Y, Burchfiel B, Tedrake R, Song S. Diffusion Policy: Visuomotor policy learning via action diffusion. *The International Journal of Robotics Research*. 2023 Jun:02783649241273668. 
- [2] Feng Z, Luan H, Goyal P, Soh H. LTLDoG: Satisfying Temporally-Extended Symbolic Constraints for Safe Diffusion-Based Planning, *IEEE Robotics and Automation Letters*, vol. 9, no. 10, pp. 8571-8578, Oct. 2024. 
