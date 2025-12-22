import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print
import matplotlib
import einx

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")


def apply_float_colormap(image: torch.Tensor, colormap: str = "viridis") -> torch.Tensor:
    """Convert single channel to a color image.

    Args:
        image: Single channel image. [b, 1, h, w]
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 3, 1, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    ret = torch.tensor(
        matplotlib.colormaps[colormap].colors,  # type: ignore
        device=image.device,
    )[image_long[:, 0, :, :]]
    return einx.rearrange("b h w c -> b c h w", ret)


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 

@torch.no_grad()
def compute_masked_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Float[Tensor, "batch height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images, with a mask.
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2 * mask[:, None], "b c h w -> b", "sum") / (mask.sum(dim=(1, 2)) + 1e-8) / 3.0 # normalize by 3 channels
    return -10 * torch.log10(mse + 1e-8)

@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    input_data, target_data = result.input, result.target
    if hasattr(result, "runtime_metrics"):
        runtime_metrics = result.runtime_metrics
    else:
        runtime_metrics = None

    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()

        # Save images
        _save_images(result, batch_idx, sample_dir)

        # Compute and save metrics if requested
        if compute_metrics:
            pc_image = None
            if target_data.get("pc_image", None) is not None:
                pc_image = target_data.pc_image[batch_idx]

            gt_image = target_data.image[batch_idx]
            pred_image = result.render[batch_idx]
            mask = None
            if target_data.get("valid_mask", None) is not None:
                mask = target_data.valid_mask[batch_idx]
                gt_image = gt_image * mask
                pred_image = pred_image * mask
            _save_metrics(
                gt_image,
                pred_image,
                target_indices,
                sample_dir,
                scene_name,
                pc_image,
                mask,
                runtime_metrics,
            )

        # Save video if available
        if hasattr(result, "video_rendering"):
            _save_video(result.video_rendering[batch_idx], sample_dir)

def visualize_intermediate_results(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    if result.render is not None:
        target_image = target.image
        rendered_image = result.render
        b, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        vis_channels = [target_image]
        if target.get("pc_image", None) is not None:
            target_pc_image = target.pc_image.reshape(b * v, -1, h, w)
            vis_channels.append(target_pc_image)
        if (
            target.get("ray_o", None) is not None
            and target.get("ray_d", None) is not None
        ):
            o_dot_d = torch.cross(target.ray_o, target.ray_d, dim=2)
            vis_plucker = (o_dot_d * 0.5 + 0.5).clamp(0, 1)
            vis_channels.append(vis_plucker)
        if result.get("conf", None) is not None:
            conf = result.conf.reshape(b * v, -1, h, w).log() # [0, infin]
            conf = conf / conf.amax()
            conf = apply_float_colormap(conf, colormap="viridis")
            vis_channels.append(conf)
        vis_channels.append(rendered_image)
        visualized_image = torch.cat(vis_channels, dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

        uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

        uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
        )
        with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
            uids = "_".join([f"{uid:08}" for uid in uids])
            f.write(uids)

    input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"

    # Create a grid of input images
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

    # Save the input image grid
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg")
    )


def _save_images(result, batch_idx, out_dir):
    """Save visualization images."""
    # Save input image
    input_img = result.input.image[batch_idx]
    input_img = rearrange(input_img, "v c h w -> h (v w) c")
    input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))

    gt_image = result.target.image[batch_idx]
    pred_image = result.render[batch_idx]
    if result.target.get("valid_mask", None) is not None:
        mask = result.target.valid_mask[batch_idx]
        gt_image = gt_image * mask
        pred_image = pred_image * mask

    # Save GT vs prediction side-by-side
    vis_channels = [gt_image]
    if result.target.get("pc_image", None) is not None:
        vis_channels.append(result.target.pc_image[batch_idx])
    if (
        result.target.get("ray_o", None) is not None
        and result.target.get("ray_d", None) is not None
    ):
        o_dot_d = torch.cross(result.target.ray_o, result.target.ray_d, dim=2)[
            batch_idx
        ]
        vis_plucker = o_dot_d * 0.5 + 0.5
        vis_channels.append(vis_plucker)
    vis_channels.append(pred_image)
    comparison = torch.cat(vis_channels, dim=2).detach().cpu()
    comparison = rearrange(comparison, "v c h w -> h (v w) c")
    comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred.png"))


def _save_metrics(
    target,
    prediction,
    view_indices,
    out_dir,
    scene_name,
    pc_image=None,
    mask=None,
    runtime_metrics=None,
):
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)

    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)
    mask = pc_image.sum(dim=1) > 1e-3 if pc_image is not None else None
    seen_psnr_values = compute_masked_psnr(target, prediction, mask) if mask is not None else None
    unseen_psnr_values = compute_masked_psnr(target, prediction, 1 - mask.float()) if mask is not None else None
    pc_psnr_values = compute_psnr(pc_image, target) if pc_image is not None else None
    pc_seen_psnr_values = compute_masked_psnr(pc_image, target, mask) if mask is not None else None
    masked_psnr_values = (
        compute_masked_psnr(target, prediction, mask) if mask is not None else None
    )

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean()),
            "seen_psnr": (
                float(seen_psnr_values.mean()) if seen_psnr_values is not None else None
            ),
            "unseen_psnr": (
                float(unseen_psnr_values.mean())
                if unseen_psnr_values is not None
                else None
            ),
            "pc_psnr": (
                float(pc_psnr_values.mean()) if pc_psnr_values is not None else None
            ),
            "pc_seen_psnr": (
                float(pc_seen_psnr_values.mean())
                if pc_seen_psnr_values is not None
                else None
            ),
            "masked_psnr": (
                float(masked_psnr_values.mean())
                if masked_psnr_values is not None
                else None
            ),
        },
        "per_view": [],
    }

    if runtime_metrics is not None:
        metrics["summary"].update(runtime_metrics)

    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append(
            {
                "view": int(view_idx),
                "psnr": float(psnr_values[i]),
                "lpips": float(lpips_values[i]),
                "ssim": float(ssim_values[i]),
                "seen_psnr": (
                    float(seen_psnr_values[i]) if seen_psnr_values is not None else 0
                ),
                "unseen_psnr": (
                    float(unseen_psnr_values[i])
                    if unseen_psnr_values is not None
                    else 0
                ),
                "pc_psnr": (
                    float(pc_psnr_values[i]) if pc_psnr_values is not None else 0
                ),
                "pc_seen_psnr": (
                    float(pc_seen_psnr_values[i])
                    if pc_seen_psnr_values is not None
                    else 0
                ),
                "masked_psnr": (
                    float(masked_psnr_values[i])
                    if masked_psnr_values is not None
                    else 0
                ),
            }
        )

    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []

    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue

        valid_subfolders.append(subfolder)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name" or metric_value is None:
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")

        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")

        f.write("\n")

        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")

    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
