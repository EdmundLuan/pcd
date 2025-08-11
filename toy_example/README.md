# 🎯 Constrained Diffusion Toy Example 

This repository contains a toy experiment to illustrate the different resulting distributions of two spheres of different sizes after applying various variations of Langevin Monte Carlo (LMC).

The problem setting is visualized using the `problem_setting.ipynb` notebook. This notebook generates plots to describe the initial setup of the experiment.


## 📂 Project Structure

```
problem_setting.ipynb    # Jupyter notebook for generating plots and visualizations
README.md                # Project documentation
requirements.txt         # Python dependencies
plots/                   # Directory containing generated plots
src/                     # Source code directory
├── lmc.py               # Implementation of LMC (Langevin Monte Carlo)
└── main.py              # Main script for running the toy example
```

</br>
</br>

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. It is recommended to use a virtual environment to manage dependencies.


### Installation

1. Install the required dependencies:
    ```bash
    conda create -n toy-example python=3.10 -y
    conda activate toy-example
    pip install -r requirements.txt
    ```


### Usage

1. Open the Jupyter notebook `problem_setting.ipynb` to generate and view visualizations:
   ```bash
   jupyter notebook problem_setting.ipynb
   ```

2. Run the main script to execute the toy example:
   ```bash
   python -m src.main
   ```


### Outputs

Generated plots can be found in the `plots/` directory. These include visualizations of the constrained diffusion processes.


### 💡 NOTES

Configurations are centralised in [src/main.py](src/main.py).