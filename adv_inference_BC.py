import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
import datetime
from model.pipeline_BC import CatVTONPipeline
import warnings
from torchvision.utils import save_image

class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.data = self.load_data()

    def load_data(self):
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, target_cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'target_cloth', 'mask']]
        ref_person_images = [self.vae_processor.preprocess(Image.open(p), self.args.height, self.args.width)[0] for p in data['ref_persons']]
        ref_mask_images = [self.mask_processor.preprocess(Image.open(m), self.args.height, self.args.width)[0] for m in data['ref_masks']]

        return {
            'index': idx,
            'person_name': data['person_name'],
            'cloth_name': data.get('cloth_name', ''),
            'target_cloth_name': data.get('target_cloth_name', ''),
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'target_cloth': self.vae_processor.preprocess(target_cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0],
            'ref_persons': torch.stack(ref_person_images),
            'ref_masks': torch.stack(ref_mask_images)

        }

class VITONHDTestDataset(InferenceDataset):
    def load_data(self):
        # Determine the complete path of the pair file
        if self.args.pair_file_path:
            # If a complete path is specified, use it directly
            pair_txt = self.args.pair_file_path
        else:
            # Otherwise use data_root_path + pair_file
            pair_txt = os.path.join(self.args.data_root_path, self.args.pair_file)
        
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
        
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
        output_dir = os.path.join(self.args.output_dir, "vitonhd", 'unpaired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            splits = line.strip().split()
            main_person = splits[0]
            ref_persons = splits[1:-2]
            cloth_img = splits[-2]
            target_cloth_img = splits[-1]
            if os.path.exists(os.path.join(output_dir, main_person)):
                continue
            if self.args.eval_pair:
                cloth_img = main_person
                target_cloth_img = main_person
            data.append({
                'person_name': main_person,
                'cloth_name': cloth_img,
                'target_cloth_name': target_cloth_img,
                'person': os.path.join(self.args.data_root_path, 'image', main_person),
                'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                'target_cloth': os.path.join(self.args.data_root_path, 'cloth', target_cloth_img),                
                'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', main_person.replace('.jpg', '_mask.png')),
                'ref_persons': [
                    os.path.join(self.args.data_root_path, 'image', p) for p in ref_persons
                ],
                'ref_masks': [
                    os.path.join(self.args.data_root_path, 'agnostic-mask', p.replace('.jpg', '_mask.png')) for p in ref_persons
                ]
            })

        return data

class DressCodeTestDataset(InferenceDataset):
    def load_data(self):
        data = []
        for sub_folder in ['upper_body', 'lower_body', 'dresses']:
            assert os.path.exists(os.path.join(self.args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
            pair_txt = os.path.join(self.args.data_root_path, sub_folder, 'test_pairs_paired.txt' if self.args.eval_pair else 'test_pairs_unpaired.txt')
            assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
            with open(pair_txt, 'r') as f:
                lines = f.readlines()

            output_dir = os.path.join(self.args.output_dir, f"dresscode-{self.args.height}",
                                      'unpaired' if not self.args.eval_pair else 'paired', sub_folder)
            for line in lines:
                person_img, cloth_img = line.strip().split(" ")
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                data.append({
                    'person_name': os.path.join(sub_folder, person_img),
                    'person': os.path.join(self.args.data_root_path, sub_folder, 'images', person_img),
                    'cloth': os.path.join(self.args.data_root_path, sub_folder, 'images', cloth_img),
                    'mask': os.path.join(self.args.data_root_path, sub_folder, 'agnostic_masks', person_img.replace('.jpg', '.png'))
                })
        return data


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="../stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="../CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vitonhd",
        help="The datasets to use for evaluation.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="../vitonhd",
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size for evaluation."
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint",
        action="store_true",
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--eval_pair",
        action="store_true",

        help="Whether or not to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        default=True,
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--concat_axis",
        type=str,
        choices=["x", "y", 'random'],
        default="y",
        help="The axis to concat the cloth feature, select from ['x', 'y', 'random'].",
    )
    parser.add_argument(
        "--enable_condition_noise",
        action="store_true",
        default=True,
        help="Whether or not to enable condition noise.",
    )
    # New parameters
    parser.add_argument(
        "--do_attack",
        action = "store_false",
        default = True,
        help="Whether to perform attack optimization",
    )
    parser.add_argument(
        "--attack_steps",
        type=int,
        default=500,#300
        help="Number of optimization steps for attack",
    )
    parser.add_argument(
        "--attack_lr",
        type=float,
        default=0.05,
        help="Learning rate for attack optimization",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.2,
        help="Weight of loss2 (condition_latent, ref_condition_latent)",
    )

    parser.add_argument(
        "--model_pool_size",
        type=int,
        default=1,
        help="Number of different person models to use in the optimization (0 for all)",
    )
    parser.add_argument(
        "--visualize_interval",
        type=int,
        default=10,
        help="Number of different person models to use in the optimization (0 for all)",
    )

    parser.add_argument(
        "--use_lpips",
        action = "store_false",
        default = True,
        help="Whether to perform attack optimization",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of steps without improvement before reducing learning rate",
    )
    parser.add_argument(
        "--pair_file",
        type=str,
        default="test_xxypairs.txt",
        help="Name of the pair file to read (e.g., test_xxypairs.txt, test_pairs.txt)",
    )
    parser.add_argument(
        "--pair_file_path",
        type=str,
        default=None,
        help="Full path to the pair file. If provided, this overrides data_root_path + pair_file",
    )

   
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def repaint(person, mask, result): # Repaint with original background
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images): # Image concatenation for easy evaluation of results
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images



def main():
    
    swanlab.init(
    # Set project name
    project="CATVTON_BC",
    logdir="./swanlog1",
    # Set hyperparameters
    )
    args = parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory name containing parameter information, first time, then other parameters
    params_info = ""
    if args.do_attack:
        params_info = f"{current_time}_k{args.k}_steps{args.attack_steps}_models{args.model_pool_size}"
    else:
        params_info = current_time

    # Pipeline
    pipeline = CatVTONPipeline(
        attn_ckpt_version=args.dataset_name,
        attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        skip_safety_check=True,

    )
    # Dataset
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTestDataset(args)
    elif args.dataset_name == "dresscode":
        dataset = DressCodeTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )

    person_pool = []
    person_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, batch in enumerate(person_loader):
        person_pool.append(batch['person'])
    
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    # Add parameter information to output directory
    args.output_dir = os.path.join(
        args.output_dir, 
        f"{args.dataset_name}-{args.height}_{params_info}", 
        "paired" if args.eval_pair else "unpaired"
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader): # Use dataloader to get data in batches, then feed this data into CatVTONPipeline model for inference
        person_images = batch['person']
        cloth_images = batch['cloth']
        target_cloth_images = batch['target_cloth']  
        masks = batch['mask']

        # Reference sample set of models for optimization
        ref_persons = batch['ref_persons']  # [B, N, C, H, W]  
        ref_masks = batch['ref_masks']      # [B, N, 1, H, W]


        if args.do_attack:

            # Store original input
            condition_image = cloth_images.clone()
            # Store original cloth image as reference for regularization
            ref_condition_image = cloth_images.clone()

            # Loop through different models to optimize condition_image, which is the cloth being optimized
            for i in range(min(args.model_pool_size, ref_persons.size(1))):
                print(f"Using model {i+1}/{min(args.model_pool_size, ref_persons.size(1))} for optimization")
                
                current_person = ref_persons[:, i, :, :, :]
                current_mask = ref_masks[:, i, :, :, :]

                # Execute attack
                condition_image,intermediate_results = pipeline.attack(
                    current_person,
                    condition_image,
                    ref_condition_image,
                    target_cloth_images,
                    current_mask,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    attack_steps=args.attack_steps,
                    attack_lr=args.attack_lr,
                    k=args.k,
                    visualize_interval = args.visualize_interval, # Visualize every n steps
                    use_lpips=args.use_lpips,  # LPIPS loss
                    patience=args.patience,
                )

                # Save intermediate results
                if not os.path.exists(os.path.join(args.output_dir, "intermediate_results")):
                    os.makedirs(os.path.join(args.output_dir, "intermediate_results"))



                # Save optimized cloth image
                save_dir = "/AdverCat/CatVTON-edited/optcloth"
                os.makedirs(save_dir, exist_ok=True)
                
                # Construct filename for each sample in the batch and save
                for batch_idx in range(condition_image.size(0)):
                    person_name = batch['person_name'][batch_idx]
                    cloth_name = batch['cloth_name'][batch_idx]
                    target_cloth_name = batch['target_cloth_name'][batch_idx]
                    
                    # Remove file extensions
                    person_base = person_name.replace('.jpg', '').replace('.png', '')
                    cloth_base = cloth_name.replace('.jpg', '').replace('.png', '')
                    target_cloth_base = target_cloth_name.replace('.jpg', '').replace('.png', '')
                    
                    # Construct filename according to required format: model{person}__ori{cloth}target{target_cloth}
                    image_name = f"model{person_base}__{person_base}ori{cloth_base}target{target_cloth_base}.png"
                    save_path = os.path.join(save_dir, image_name)
                    
                    # Save the optimized image for this sample
                    save_image(condition_image[batch_idx:batch_idx+1], save_path)
                # Create visualization images
                for result in intermediate_results:
                    step = result['step']
                    condition_img = to_pil_image(result['condition_image'])[0]  # Convert tensor to PIL image
                    result_img0 = result['result'][0][0]  # First PIL image in the first result list
                    result_img1 = result['result'][1][0]  # First PIL image in the second result list
                    
                    # Create concatenated image containing condition image and two results
                    w, h = result_img0.size
                    concat_img = Image.new('RGB', (w*3, h))
                    concat_img.paste(condition_img, (0, 0))
                    concat_img.paste(result_img0, (w, 0))
                    concat_img.paste(result_img1, (w*2, 0))
                    
                    # Save concatenated image
                    output_path = os.path.join(
                        args.output_dir, 
                        "intermediate_results", 
                        f"step{step:04d}.png"
                    )
                    concat_img.save(output_path)

            # Use optimized condition image for final inference
            results = pipeline(
                person_images,                
                condition_image,
                target_cloth_images,
                masks,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
                ret_latent=False,
            )

        else:
            # Original inference process
            results = pipeline(
                person_images,                
                cloth_images,
                target_cloth_images,
                masks,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
                ret_latent=False  
            )

        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images)
            cloth_images = to_pil_image(cloth_images)
            target_cloth_images = to_pil_image(target_cloth_images) 
            masks = to_pil_image(masks)

        # Process each result image
        for i, (result0, result1) in enumerate(zip(results[0], results[1])):  # Unpack two results
            person_name = batch['person_name'][i]

            # Add parameter information to filename, first time, then other parameters
            if args.do_attack:
                file_suffix = f"_{current_time}_k{args.k}_steps{args.attack_steps}_models{args.model_pool_size}"
            else:
                file_suffix = f"_{current_time}"

            # Create different output paths for two results
            output_path0 = os.path.join(args.output_dir, f"{os.path.splitext(person_name)[0]}_result0{file_suffix}.png")
            output_path1 = os.path.join(args.output_dir, f"{os.path.splitext(person_name)[0]}_result1{file_suffix}.png")
            
            # Ensure output directory exists
            if not os.path.exists(os.path.dirname(output_path0)):
                os.makedirs(os.path.dirname(output_path0))
                
            # If repaint is needed
            if args.repaint:
                person_path = dataset.data[batch['index'][i]]['person']
                mask_path = dataset.data[batch['index'][i]]['mask']
                person_image = Image.open(person_path).resize(result0.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result0.size, Image.NEAREST)
                result0 = repaint(person_image, mask, result0)
                result1 = repaint(person_image, mask, result1)
                
            # If concatenation of results is needed
            if args.concat_eval_results:
                w, h = result0.size
                if args.do_attack:
                    
                    # Save optimized condition image
                    # Save optimized condition image with parameter information in filename
                    optimize_path = os.path.join(args.output_dir, f"{os.path.splitext(person_name)[0]}_optimized{file_suffix}.png")
                    optimize_image = to_pil_image(condition_image)[i]  # Convert tensor to PIL image
                    optimize_image.save(optimize_path)
                
                    # In attack mode, display 6 images
                    concated_result = Image.new('RGB', (w*6, h))
                    concated_result.paste(person_images[i], (0, 0))
                    concated_result.paste(cloth_images[i], (w, 0))
                    concated_result.paste(optimize_image, (w*2, 0))
                    concated_result.paste(target_cloth_images[i], (w*3, 0))
                    concated_result.paste(result1, (w*4, 0))
                    concated_result.paste(result0, (w*5, 0))
                    concated_result.save(output_path0)

                else:
                    # In normal mode, can create two concatenated results (each containing one output from the model)
                    concated_result0 = Image.new('RGB', (w*3, h))
                    concated_result0.paste(person_images[i], (0, 0))
                    concated_result0.paste(cloth_images[i], (w, 0))
                    concated_result0.paste(result0, (w*2, 0))
                    concated_result0.save(output_path0)
                    
                    concated_result1 = Image.new('RGB', (w*3, h))
                    concated_result1.paste(person_images[i], (0, 0))
                    concated_result1.paste(target_cloth_images[i], (w, 0))
                    concated_result1.paste(result1, (w*2, 0))
                    concated_result1.save(output_path1)
            else:
                # If not concatenating, save two results separately
                result0.save(output_path0)
                result1.save(output_path1)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
