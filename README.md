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

1. **Environment and Data Setup**
   This project is built upon the original CatVTON. Before running the attack, you **must** first set up the complete environment as described in the official [CatVTON repository](https://github.com/Zheng-Chong/CatVTON). This includes:
    - Creating the `catvton` conda environment.
    - Downloading all required pre-trained models (Stable Diffusion, VAE, CatVTON weights).
    - Preparing the VITON-HD or DressCode datasets.

2. **Place Attack-Specific Files**
   - Place the `pair_mutiperson1.txt` file into the root of your prepared VITON-HD dataset directory (e.g., `/path/to/your/vitonhd_dataset/`).
   - Place the target cloth image (e.g., `000_00.jpg`) into the cloth testing directory (e.g., `/path/to/your/vitonhd_dataset/test/cloth/`).

3. **Install Additional Dependencies**
   After setting up the CatVTON environment, install a few extra packages required for this project:
   ```bash
   conda activate catvton
   pip install swanlab pyiqa open_clip-torch
   ```

4. **Run Adversarial Attack**
   Once the environment is ready, run the attack script. Ensure the paths point to the models and data you prepared in Step 1.
   ```bash
   python adv_inference_mutiperson.py \
     --resume_path /path/to/your/CatVTON/weights \
     --data_root_path /path/to/your/vitonhd_dataset \
     --output_dir ./output \
     --do_attack \
     --attack_steps 500 \
     --attack_lr 0.05 \
     --k 0.2
   ```
   See `parse_args()` in the script for all configurable options.

### Acknowledgement

This code is modified from [CatVTON](https://github.com/Zheng-Chong/CatVTON) and is intended for academic research and security analysis only.