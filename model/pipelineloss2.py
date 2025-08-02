import inspect
import os
from typing import Union
import random
from torch import optim
from tqdm import tqdm
import pyiqa
import PIL
from PIL import Image
from torchvision.utils import save_image
import numpy as np
import torch
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor
import lpips

import pyiqa
import open_clip
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (
    compute_vae_encodings,
    numpy_to_pil,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)


class CatVTONPipeline:
    def __init__(
        self,
        base_ckpt,
        attn_ckpt,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cuda",
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check
        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("./sd-vae-ft-mse").to(device, dtype=weight_dtype)
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                base_ckpt, subfolder="safety_checker"
            ).to(device, dtype=weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        self.lpips_model = lpips.LPIPS(net="vgg").to(device)  # Initialize LPIPS model
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        # Enable gradient checkpointing to save memory and prevent gradient explosion
        self.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, "attention"))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, "attention"))

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept

    def check_inputs(self, image, condition_image, target_condition_image, mask, width, height):
        if (
            isinstance(image, torch.Tensor)
            and isinstance(condition_image, torch.Tensor)
            and isinstance(target_condition_image, torch.Tensor)
            and isinstance(mask, torch.Tensor)
        ):
            return image, condition_image, target_condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        target_condition_image = resize_and_padding(target_condition_image, (width, height))
        return image, condition_image, target_condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # @staticmethod
    # def get_k(self,t, t_min=1, t_max=981, k_min=1, k_max=None):
    #     if k_max is None:
    #         k_max = self.kmax

    #     if isinstance(t, torch.Tensor):
    #         t = t.item()

    #     t_normalized = (t - t_min) / (t_max - t_min)
    #     beta = self.beta                 # Originally was 3.0
    #     k = k_min + (k_max - k_min) * (np.exp(-beta * t_normalized) - np.exp(-beta)) / (1 - np.exp(-beta))

    #     return k

    def _run_diffusion_with_grad(
        self,
        masked_latent,
        condition_latent,
        mask_latent,
        num_inference_steps,
        guidance_scale,
        generator,
        eta,
        concat_dim,
        fixed_noise=None,
    ):

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        # Prepare noise
        latents = fixed_noise.clone()

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop with simpler gradient checkpointing strategy
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Define single-step denoising function to avoid closure capturing external variables
        def denoise_step(
            latents_input, t, masked_latent_concat, mask_latent_concat, do_classifier_free_guidance, guidance_scale
        ):
            """Single-step denoising function, all dependencies passed as parameters"""

            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents_input] * 2) if do_classifier_free_guidance else latents_input
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # Prepare input
            x = torch.cat([latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)

            # Predict noise
            noise_pred = self.unet(
                x,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            # Apply guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents_input, **extra_step_kwargs).prev_sample
            return latents

        progress_bar = tqdm(total=len(timesteps), desc="Diffusion Progress", ncols=80)

        for i, t in enumerate(timesteps):
            # Use gradient checkpointing
            latents = torch.utils.checkpoint.checkpoint(
                denoise_step,
                latents,
                t,
                masked_latent_concat,
                mask_latent_concat,
                do_classifier_free_guidance,
                guidance_scale,
                use_reentrant=False,
            )
            progress_bar.update(1)

        progress_bar.close()
        # Return final tensor
        final_latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        # final_latents = 1 / self.vae.config.scaling_factor * final_latents

        return final_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        target_condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        ret_latent=False,
        **kwargs,
    ):
        concat_dim = -2  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image, target_condition_image, mask = self.check_inputs(
            image, condition_image, target_condition_image, mask, width, height
        )
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        target_condition_image = prepare_image(target_condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        target_condition_latent = compute_vae_encodings(target_condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image, target_condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        target_masked_latent_concat = torch.cat([masked_latent, target_condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps  # Get timestep sequence
        latents = (
            latents * self.noise_scheduler.init_noise_sigma
        )  # Scale initial noise self.noise_scheduler.init_noise_sigma = 1
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )

            target_masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(target_condition_latent)], dim=concat_dim),
                    target_masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Copy noise into two parts for separate diffusion
        latents0 = latents.clone()
        latents1 = None if ret_latent else latents.clone()

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order  # order = 1

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                original_latent0 = torch.cat([latents0] * 2) if do_classifier_free_guidance else latents0
                original_latent0 = self.noise_scheduler.scale_model_input(original_latent0, t)
                # prepare the input for the inpainting model
                x0 = torch.cat([original_latent0, mask_latent_concat, target_masked_latent_concat], dim=1)
                # predict the noise residual
                noise_pred0 = self.unet(
                    x0,
                    t.to(self.device),
                    encoder_hidden_states=None,  # FIXME
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond0, noise_pred_text0 = noise_pred0.chunk(2)
                    noise_pred0 = noise_pred_uncond0 + guidance_scale * (noise_pred_text0 - noise_pred_uncond0)
                latents0 = self.noise_scheduler.step(noise_pred0, t, latents0, **extra_step_kwargs).prev_sample
                # latents1 is generated using original image and model combination, not used in optimization iteration
                if latents1 is not None:
                    original_latent1 = torch.cat([latents1] * 2) if do_classifier_free_guidance else latents1
                    original_latent1 = self.noise_scheduler.scale_model_input(original_latent1, t)
                    x1 = torch.cat([original_latent1, mask_latent_concat, masked_latent_concat], dim=1)
                    noise_pred1 = self.unet(
                        x1,
                        t.to(self.device),
                        encoder_hidden_states=None,  # FIXME
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond1, noise_pred_text1 = noise_pred1.chunk(2)
                        noise_pred1 = noise_pred_uncond1 + guidance_scale * (noise_pred_text1 - noise_pred_uncond1)
                    latents1 = self.noise_scheduler.step(noise_pred1, t, latents1, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        if ret_latent:
            return latents0

        # Process final latent vectors to get final image
        # Decode the final latents
        latents0 = latents0.split(latents0.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents0 = 1 / self.vae.config.scaling_factor * latents0
        image0 = self.vae.decode(latents0.to(self.device, dtype=self.weight_dtype)).sample
        image0 = (image0 / 2 + 0.5).clamp(0, 1)
        latents1 = latents1.split(latents1.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents1 = 1 / self.vae.config.scaling_factor * latents1
        image1 = self.vae.decode(latents1.to(self.device, dtype=self.weight_dtype)).sample
        image1 = (image1 / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image0 = image0.cpu().permute(0, 2, 3, 1).float().numpy()
        image0 = numpy_to_pil(image0)
        image1 = image1.cpu().permute(0, 2, 3, 1).float().numpy()
        image1 = numpy_to_pil(image1)

        return image0, image1  # 0 is target, 1 is original

    def attack(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        ref_condition_image: Union[PIL.Image.Image, torch.Tensor],  # Original unoptimized condition_image for regularization term
        target_condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        attack_steps: int = 100,
        attack_lr: float = 0.1,
        k: float = 0.025,
        eta=1.0,
        visualize_interval: int = 50,  # Control visualization interval
        use_lpips: bool = True,  # Whether to use LPIPS loss
        patience: int = 10,  # New: reduce learning rate after how many steps of stagnation
        **kwargs,
    ):

        concat_dim = -2  # FIXME: y axis concat

        # Prepare inputs to Tensor
        image, condition_image, target_condition_image, mask = self.check_inputs(
            image, condition_image, target_condition_image, mask, width, height
        )
        if not isinstance(ref_condition_image, torch.Tensor):
            ref_condition_image = resize_and_padding(ref_condition_image, (width, height))

        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        target_condition_image = prepare_image(target_condition_image).to(self.device, dtype=self.weight_dtype)
        ref_condition_image = prepare_image(ref_condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        # VAE encoding
        masked_image = image * (mask < 0.5)
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        target_condition_latent = compute_vae_encodings(target_condition_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae).clone().detach()
        ref_condition_latent = compute_vae_encodings(ref_condition_image, self.vae).clone().detach()

        condition_latent.requires_grad_(True)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

        # Set up Adam optimizer
        optimizer = optim.Adam([condition_latent], attack_lr)
        #optimizer = optim.AdamW([condition_latent], lr=attack_lr, weight_decay=0.01)  # Weight decay parameter is adjustable

        intermediate_results = []
        # Generate fixed noise to ensure all diffusion processes use the same starting noise
        masked_latent_concat = torch.cat([masked_latent, target_condition_latent], dim=concat_dim)
        fixed_noise = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        iqa_metric = pyiqa.create_metric('lpips', device='cuda', as_loss=True)

        print(f"Initial ref_latent: requires_grad={condition_latent.requires_grad},min={condition_latent.min().item():.6f}, max={condition_latent.max().item():.6f},norm={condition_latent.norm().item():.6f}")

        # Pre-compute target result (only needs to be computed once)
        target_result_latent = self._run_diffusion_with_grad(
            masked_latent,
            target_condition_latent,
            mask_latent,
            num_inference_steps,
            guidance_scale,
            generator,
            eta,
            concat_dim,
            fixed_noise=fixed_noise,
        ).detach()  # Detach computation graph to avoid unnecessary gradient computation

        print(f"Target result calculation completed, starting optimization loop...")
        lpips_metric = pyiqa.create_metric('lpips', device='cuda', as_loss=True)
        ssim_metric = pyiqa.create_metric('ssim', device='cuda', as_loss=True)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(self.device)
        def calc_clip_similarity(img_tensor1, img_tensor2):
            # Convert tensors to PIL images
            def tensor_to_pil(tensor):
                # Ensure tensor is on CPU and float32
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                if tensor.dtype != torch.float32:
                    tensor = tensor.float()
                # Detach gradients so numpy() can be called
                tensor = tensor.detach()
                # Convert to numpy and adjust dimension order
                img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
                # Ensure values are in [0,1] range, then convert to [0,255]
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                return Image.fromarray(img_np)
            
            pil_img1 = tensor_to_pil(img_tensor1)
            pil_img2 = tensor_to_pil(img_tensor2)
            
            img1 = clip_preprocess(pil_img1).unsqueeze(0).to(self.device)
            img2 = clip_preprocess(pil_img2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat1 = clip_model.encode_image(img1)
                feat2 = clip_model.encode_image(img2)
                feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
                feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
                sim = (feat1 @ feat2.T).item()
            return sim
        # Optimization loop
        for i in range(attack_steps):

            with torch.enable_grad():
                result_latent = self._run_diffusion_with_grad(
                    masked_latent,
                    condition_latent,
                    mask_latent,
                    num_inference_steps,
                    guidance_scale,
                    generator,
                    eta,
                    concat_dim,
                    fixed_noise=fixed_noise,
                )

            cos_sim1 = torch.nn.functional.cosine_similarity(target_result_latent, result_latent)
            loss1 = (1 - cos_sim1).mean() 


            result_latent0 = 1 / self.vae.config.scaling_factor * result_latent
            result_image0 = self.vae.decode(result_latent0.to(self.device, dtype=self.weight_dtype)).sample
            result_image0 = (result_image0 / 2 + 0.5).clamp(0, 1)
            
            condition_latent0 = 1 / self.vae.config.scaling_factor * condition_latent
            condition_image0 = self.vae.decode(condition_latent0.to(self.device, dtype=self.weight_dtype)).sample
            condition_image0 = (condition_image0 / 2 + 0.5).clamp(0, 1)
            
            target_condition_latent0 = 1 / self.vae.config.scaling_factor * target_condition_latent
            target_condition_image0 = self.vae.decode(target_condition_latent0.to(self.device, dtype=self.weight_dtype)).sample
            target_condition_image0 = (target_condition_image0 / 2 + 0.5).clamp(0, 1)

            ref_condition_latent0 = 1 / self.vae.config.scaling_factor * ref_condition_latent
            ref_condition_image0 = self.vae.decode(ref_condition_latent0.to(self.device, dtype=self.weight_dtype)).sample
            ref_condition_image0 = (ref_condition_image0 / 2 + 0.5).clamp(0, 1)
            loss2 =iqa_metric(condition_image0, ref_condition_image0)
           
            result_latent_target = 1 / self.vae.config.scaling_factor * target_result_latent
            targetresult_image0 = self.vae.decode(result_latent_target.to(self.device, dtype=self.weight_dtype)).sample
            targetresult_image0 = (targetresult_image0 / 2 + 0.5).clamp(0, 1)

            LPIPS_targetcloth_vs_origincloth= lpips_metric(target_condition_image, ref_condition_image0)

            LPIPS_optcloth_vs_origincloth = lpips_metric(condition_image0, ref_condition_image0)
            SSIM_optcloth_vs_origincloth = ssim_metric(condition_image0, ref_condition_image0)
            SSIM_targetcloth_vs_origincloth = ssim_metric(target_condition_image0, ref_condition_image0)

            LPIPS_optresult_vs_targetresult= lpips_metric(result_image0, targetresult_image0)
            SSIM_optresult_vs_targetresult=ssim_metric(result_image0, targetresult_image0)
          
            clip_opt_result = calc_clip_similarity(result_image0, targetresult_image0)
            # Total loss SSIM
            total_loss = loss1 + loss2 * k
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([condition_latent], max_norm=1.0)  

            optimizer.step()

            print(f"Step {i}/{attack_steps}: Total loss = {total_loss.item():.6f}, Similarity loss = {loss1.item():.6f}, Regularization loss = {loss2.item():.6f}")

        
       

        with torch.no_grad():
            # Decode the final latents
            condition_latent = 1 / self.vae.config.scaling_factor * condition_latent
            optimized_condition_image = self.vae.decode(
                condition_latent.to(self.device, dtype=self.weight_dtype)
            ).sample
        return optimized_condition_image,intermediate_results
