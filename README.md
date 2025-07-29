## Mutiperson Adversarial Attack for CatVTON

This repository contains an adversarial attack implementation for CatVTON, focusing on the multi-person (mutiperson) setting. The code is based on the original CatVTON framework, with modifications to support adversarial optimization targeting the virtual try-on model.

### Overview

- **Purpose:**  
  To evaluate and attack the robustness of CatVTON under multi-person conditions by generating adversarial examples that can mislead the model's output.

- **Key Features:**  
  - Supports multi-person reference images and masks for adversarial optimization.
  - Flexible attack configuration: number of steps, learning rate, loss weights, etc.
  - Compatible with VITONHD and DressCode datasets.
  - Visualization and saving of intermediate and final adversarial results.

### Main Files

- `adv_inference_mutiperson.py`: Main script for running adversarial attacks in the multi-person setting.
- `model/pipeline_mutiperson.py`: Modified CatVTON pipeline for multi-person adversarial inference.

### Usage

1. **Install dependencies**
   ```bash
   pip install torch diffusers pillow tqdm swanlab
   ```

2. **Prepare model and data**
   - Download CatVTON pretrained weights and place them in the directory specified by `--resume_path`.
   - Prepare VITONHD or DressCode datasets and set `--data_root_path` accordingly.

3. **Run adversarial attack**
   ```bash
   python adv_inference_mutiperson.py \
     --data_root_path /path/to/vitonhd \
     --attack_steps 500 \
     --attack_lr 0.05 \
     --k 0.2
   ```
   See `parse_args()` in the script for all configurable options.

4. **Results**
   - Adversarial images and visualizations will be saved in the output directory, with filenames indicating the attack parameters.

### Applications

- Security and robustness analysis of virtual try-on models.
- Research on adversarial examples in the context of fashion and computer vision.
- Benchmarking multi-person robustness for generative models.

### Acknowledgement

This code is modified from [CatVTON](https://github.com/Zheng-Chong/CatVTON) and is intended for academic research and security