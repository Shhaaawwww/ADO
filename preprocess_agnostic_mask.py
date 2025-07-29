import argparse
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

from model.cloth_masker import AutoMasker


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of Preprocess Agnostic Mask")
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path or repo name of CatVTON. "
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    # args.repo_path = snapshot_download(repo_id=args.repo_path)
    data_root_path = "../vitonhd"
    repo_path = "../CatVTON"
    output_dir = "../vitonhd_tmp"

    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cpu', 
    )
    pair_txt = os.path.join(data_root_path, 'test_pairs.txt')
    with open(pair_txt, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        person_img, _ = line.strip().split(" ")
        if os.path.exists(os.path.join(output_dir, person_img.replace('.jpg', '.png'))):
            continue
        mask = automasker(
            os.path.join(data_root_path, "test", 'image', person_img),
            'upper'
        )['mask']
        mask.save(os.path.join(output_dir, person_img.replace('.jpg', '_mask.png')))

    # for sub_folder in ['upper_body', 'lower_body', 'dresses']:
    #     assert os.path.exists(os.path.join(args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
    #     pair_txt = os.path.join(args.data_root_path, sub_folder, 'test_pairs_paired.txt')
    #     assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
    #     cloth_type = {'upper_body': 'upper', 'lower_body': 'lower', 'dresses': 'overall'}[sub_folder]
    #     with open(pair_txt, 'r') as f:
    #         lines = f.readlines()
    #     output_dir = os.path.join(args.data_root_path, sub_folder, 'agnostic_masks')
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     for line in tqdm(lines, desc=f"Processing {sub_folder}"):
    #         person_img, _ = line.strip().split(" ")
    #         if os.path.exists(os.path.join(output_dir, person_img.replace('.jpg', '.png'))):
    #             continue
    #         mask = automasker(
    #             os.path.join(args.data_root_path, sub_folder, 'images', person_img),
    #             cloth_type
    #         )['mask']
    #         mask.save(os.path.join(output_dir, person_img.replace('.jpg', '.png')))

if __name__ == "__main__":
    # args = parse_args()
    main()
