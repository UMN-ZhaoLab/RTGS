# RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17013198.svg)](https://doi.org/10.5281/zenodo.17013198)

A real-time Gaussian Splatting implementation for SLAM systems, building upon the excellent work of MonoGS and Photo-SLAM.

**This is the ONX (Edge GPU) version of MonoRTGS.** If you want to run on RTX or A100 GPUs, please refer to the original repository: [https://github.com/Nemo0412/MonoRTGS.git](https://github.com/Nemo0412/MonoRTGS.git)

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nemo0412/MonoRTGS_ONX.git
cd MonoRTGS_ONX
```

### 2. Setup the environment

> **⚠️ Important Note for ONX Edge GPU Users**
> 
> The `environment.yml` file is provided for reference only. On ONX edge GPUs, many packages need to be installed manually due to ARM64 architecture compatibility issues.
> 
> For PyTorch and related packages on Jetson devices, please refer to the [NVIDIA Developer Forums - PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048?utm_source=chatgpt.com) to find the appropriate pre-built wheels for your specific JetPack version and Python version.
> 
**Recommended approach for ONX:**
```bash
# Create a new conda environment
conda create -n MonoRTGS_ONX python=3.10
conda activate MonoRTGS_ONX

# Install PyTorch manually using NVIDIA's pre-built wheels
# Example for JetPack 6.0 (adjust version as needed):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies manually
pip install numpy opencv-python matplotlib scipy
```

## 🚀 Usage

### 1. SLAM Execution (run_slam.sh)

The `run_slam.sh` script allows you to run both baseline and MonoRTGS versions of the SLAM system:

```bash
./run_slam.sh <version> <config_file>
```

#### Parameters:
- **version**: `baseline` or `monortgs`
- **config_file**: Path to the configuration file

#### Examples:

```bash
# Run baseline version with TUM dataset
./run_slam.sh baseline configs/rgbd/tum/fr1_desk.yaml

# Run MonoRTGS version with TUM dataset  
./run_slam.sh monortgs configs/rgbd/tum/fr3_office.yaml

# Run baseline version with Replica dataset
./run_slam.sh baseline configs/rgbd/replica/office0.yaml

# Run MonoRTGS version with Replica dataset
./run_slam.sh monortgs configs/rgbd/replica/office1.yaml
```

### 2. Hardware Evaluation (run_hardware_eval.sh)

The `run_hardware_eval.sh` script runs the RTGS hardware speedup evaluation on ONX edge GPU:

```bash
./run_hardware_eval.sh
```

This script will:
- Activate the Python virtual environment (`~/venvs/jp6torch`)
- Check for the point cloud file
- Run the evaluation with the configured point cloud path
- Display detailed speedup analysis results

#### Configuration:

To use a different point cloud file, edit the `POINT_CLOUD_FILE` variable in `run_hardware_eval.sh`:

```bash
POINT_CLOUD_FILE="/path/to/your/point_cloud.json"
```

## 📁 Project Structure

```
MonoRTGS_ONX_demo/
├── run_slam.sh                     # SLAM execution script (baseline vs MonoRTGS)
├── run_hardware_eval.sh           # Hardware evaluation script for ONX edge GPU
├── environment.yml                 # Conda environment configuration
├── Baseline/                       # Baseline SLAM implementation
│   ├── slam.py                     # Main SLAM script
│   ├── configs/                    # Configuration files
│   │   ├── rgbd/                   # RGB-D camera configurations
│   │   │   ├── tum/                # TUM dataset configs
│   │   │   └── replica/            # Replica dataset configs
│   │   ├── stereo/                 # Stereo camera configurations
│   │   ├── mono/                   # Monocular camera configurations
│   │   └── live/                   # Live camera configurations
│   └── gaussian_splatting/         # Gaussian splatting implementation
├── MonoRTGS/                       # MonoRTGS implementation
│   ├── slam.py                     # Main SLAM script
│   ├── configs/                    # Configuration files
│   └── gaussian_splatting/         # RTGS implementation
└── hardware_speedup_simulator/     # ONX edge GPU hardware simulation
    ├── eval.py                     # Hardware speedup evaluation script
    ├── RTGS_simulator.py          # RTGS simulator for edge GPU
    ├── transform.py                # Point cloud transformation utilities
    ├── point_cloud.json           # Sample point cloud data
    └── transformed_data.json      # Transformed data for simulator
```


## 🐳 Docker Installation & Usage

You can also run this project directly using Docker, without manual environment setup:

```bash
docker pull mugen0412/monortgs:cuda12.1
docker run --rm -it --gpus all mugen0412/monortgs:cuda12.1 bash
```

## 🔗 Related Projects

### Original MonoRTGS (for RTX/A100 GPUs)

For running on high-end GPUs like RTX or A100, please check out the original repository:
- [MonoRTGS Repository](https://github.com/Nemo0412/MonoRTGS.git)

### PhotoSLAM RTGS Implementation

For a RTGS-SLAM implementation based on Photo-SLAM, please check out:
- PhotoSLAM RTGS Implementation

## 🙏 Acknowledgements

This project builds upon the excellent work of the authors of **MonoGS**, **Photo-SLAM**, and **GPGPU-Sim**.  
We gratefully acknowledge their open-source contributions, which make this project possible.

- **MonoGS** ([CVPR 2024 Highlight & Best Demo Award](https://github.com/muskie82/MonoGS.git)) - Gaussian Splatting SLAM
- **Photo-SLAM** ([CVPR 2024](https://github.com/Nemo0412/MonoRTGS.git)) - RTGS implementation
- **GPGPU-Sim** ([https://github.com/gpgpu-sim/gpgpu-sim_distribution.git](https://github.com/gpgpu-sim/gpgpu-sim_distribution.git)) - GPU architecture simulation framework that provides detailed simulation models of contemporary NVIDIA GPUs




