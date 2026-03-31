# Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2603.06043-b31b1b.svg)](https://arxiv.org/abs/2603.06043)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official open-source implementation for the paper **"Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models"**.

## 📖 Introduction

This project introduces the **GvU (Generate via Understanding)** framework. This framework aims to further enhance the generation and alignment capabilities of Unified Multimodal Models through an "Understanding-Driven Intrinsic Rewarding" mechanism.

## ⚙️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage the virtual environment. Please follow the steps below to install:

```bash
# 1. Clone this repository to your local machine
git clone [https://github.com/matrix0721/GvU.git](https://github.com/matrix0721/GvU.git)
cd GvU

# 2. Create the environment using environment.yml
conda env create -f environment.yml

# 3. Activate the virtual environment (replace <your_env_name> with the actual name in the yaml file)
conda activate <your_env_name>
```

## 🚀 Quick Start

### 1. Inference

If you want to use the pre-trained model for batch generation or testing, you can run the following inference script:

```bash
bash generate/batch_generate.sh
```
*(Note: Before running, please ensure you have configured the correct dataset paths and model weight paths within the script.)*

### 2. Training

This project utilizes methods related to GRPO (Group Relative Policy Optimization) for training. You can initiate the training process with the following script:

```bash
bash omni_r1/scripts/run_grpo.sh
```

## 📂 Project Structure

- **`evaluation/`**: Contains evaluation code and scripts for assessing the model's generation quality and multimodal understanding capabilities.
- **`generate/`**: Contains scripts and logic for model inference and batch generation.
- **`omni_r1/`**: Contains the core training framework and reinforcement learning (e.g., GRPO) scripts.
- **`environment.yml`**: Conda dependency environment configuration file.

## 📝 Citation

If you find our code or paper helpful for your research, please consider citing our work:

```bibtex
@article{pan2026learning,
  title={Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models},
  author={Jiadong Pan and Liang Li and Yuxin Peng and Yu-Ming Tang and Shuohuan Wang and Yu Sun and Hua Wu and Qingming Huang and Haifeng Wang},
  journal={arXiv preprint arXiv:2603.06043},
  year={2026},
}
```
