# PCD Implementation on MMD Benchmark

This implementation is built upon [MMD](https://github.com/yoraish/mmd) [1]. 

## What This Repo Adds/Modifies

This repo extends the upstream MMD release with:
- PCD as a planning method
- projection and coupling-cost logic under `mmd/models/projection/` and `mmd/coupling_costs/`
- inference launchers in `scripts/inference/`
- result analysis scripts in `scripts/analyze/`
- Some patches for certain files in `deps/torch_robotics/`

If you are reproducing results, use the our entrypoints detailed below rather than the upstream examples alone. 

---

## Installation

Follow the setup steps from MMD as below. 

### Requirements
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

### Steps

1. Create a conda environment and activate it:
    ```sh
    conda env create -f environment.yml
    conda activate pcdmmd
    ```
2. Install PyTorch. This may be different depending on your system. We used the following setup:
    ```sh
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
    ```
3. Install some local packages (modified version of [2, 3]):  
    ```sh
    cd deps/torch_robotics
    pip install -e .
    cd ../experiment_launcher
    pip install -e .
    cd ../motion_planning_baselines
    pip install -e .
    cd ../..
    ```
4. Install the `mmd` package:
    ```sh
    pip install -e .
    ```


### Get Datasets

Download the sample datasets and models provided by MMD. 

```sh
conda activate pcdmmd
```

```sh
gdown --id 1Onw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
tar -xJvf data_trajectories.tar.xz

gdown --id 1WO3tpvg-HU0m9RyDvGyfDamo7roBYMud
tar -xJvf data_trained_models.tar.xz
```

### Generating a Missing Dataset

Dataset for `DropRegion` is missing from the official release of MMD. 
Do the following to generate a few (all we need is some confiuration files actually):

```sh 
conda activate pcdmmd

python scripts/generate_data/launch_generate_trajectories.py
```

- The generated data will be saved in the `logs/` directory. In order to use the generated data, please move it to the `data_trajectories/` directory. Rename and reorganize the folder structure following other environments therein. 


## Planning with PCD-MMD

### Running Inference with PCD/PD/Coupling

Launch planning by executing `scripts/inference/launch_pcdiff.py`. 
***Users need to check and adjust settings in the script***. 

```bash
conda activate pcdmmd
python scripts/inference/launch_pcdiff.py --config pcdiff_empty.yaml -g coupling -p max_vel_admm -c "hinge_sqr_l2"  -r  --gpu-id 0  -m custom_comments
```
Usage: 
- `--config`: to designate config file for the task to run. See all under `configs/`. 
- `-g`: use coupling or not, should be one of `{'coupling', 'vanilla'}`. 
- `-p`: use projection or not, should be one of `{'max_vel_admm', 'none'}`. 
- `-c`: specify the cost function should be one of `{'hinge_sqr_l2', 'sum_log_l2'}`, corresponding to the SHD and LB cost specified in the paper, respectively. 
- `-r`: Cartesian product of all specified hyperparameters for multiple runs. 
- `--gpu_id`: specify CUDA device number. 

Results will be saved under the `results/` directory:  
- A log will be saved under `results/000event_logs`, containing result directories of (potentially multiple) experiments. 

### Process and Analyze Results 

1. Edit the config file for analysis `configs/analysis_config.yaml`: 
    - Paste the relative directory to the log file `xxx.log` under the `log_paths` field. 
    - Configure other switches and default constants if needed: 
        - E.g., `DEFAULT_MAX_VEL` specifies the value of max-velocity constraint; *this only affects methods that are NOT configured with such info*, e.g., vanilla diffusion or coupling-only method. The analysis script automatically retrieves the velocity constraint used for PCD or projection-only method. 
2. Run the analysis script `scripts/analyze/analyzing_pcd.py` to calculate all the metrics: 
    ```bash
    conda activate pcdmmd
    python scripts/analyze/analyzing_pcd.py --config analysis_config.yaml 
    ```
    - A JSON file will be generated and placed under each directory specified in the log file; this JSON file contains all the metrics. 
    - You need to run the analysis script multiple times if you are calculating velocity constraint satisfaction rates for vanilla diffusion or coupling-only method *for different velocity thresholds*. 

## Compare with MMD 

### Launching MMD 

1. Configure the task to run and configure number of trials in `scripts/inference/launch_mmd.py`. 
2. Run the script: 
    ```bash
    conda activate pcdmmd
    python scripts/inference/launch_mmd.py 
    ```

### Postprocess and Analyze MMD Results 

1. Configure the script `scripts/analyze/process_mmd.py` following the examples therein. 
2. Run the postprocessing script for MMD: 
    ```bash
    python scripts/analyze/process_mmd.py 
    ```
    - A dummy log file will be created under *the **parent** directory of the **first** experiment directory specified in the script*, with a timestamp in its file name. 
3. Configure the analysis config `configs/analyze_mmd.yaml` similar to the above-mentioned process for PCD. 
4. Run the analysis script: 
    ```bash
    python scripts/analyze/analyzing_mmd.py --config analyze_mmd.yaml
    ```
    - Similarly, a JSON file will be saved under the directory of each experiment run. 


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

- [1] Shaoul Y, Mishani I, Vats S, Li J, Likhachev M. Multi-robot motion planning with diffusion models. ICLR. 2025. 
- [2] Carvalho J, Le AT, Baierl M, Koert D, Peters J. Motion planning diffusion: Learning and planning of robot motions with diffusion models. 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2023 Oct 1 (pp. 1916-1923). 
- [3] Le AT, Chalvatzaki G, Biess A, Peters JR. Accelerating motion planning via optimal transport. Advances in Neural Information Processing Systems. 2023 Dec 15;36:78453-82. 

