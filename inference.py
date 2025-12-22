import importlib
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import hydra
from rich import print
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------- Dist helpers --------------------
def _set_sane_env_defaults():
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")


def is_dist():
    return dist.is_available() and dist.is_initialized()


# -------------------- Main --------------------
@hydra.main(version_base=None, config_path="./configs")
def main(config):
    _set_sane_env_defaults()

    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    print(config)

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    # --- DDP init ---
    ddp_info = init_distributed(seed=777)
    if is_dist():
        dist.barrier()

    # --- TF32 / AMP policy ---
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.training.use_tf32)
    amp_dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }

    # --- Dataset / DataLoader ---
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config, split="val")

    datasampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False,
        sampler=datasampler,
    )
    if is_dist():
        datasampler.set_epoch(0)

    if is_dist():
        dist.barrier()

    # --- Model ---
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    model = DDP(model, device_ids=[ddp_info.local_rank])
    print(f"Loading checkpoint from: {config.training.checkpoint_dir}")
    model.module.load_ckpt(config.training.checkpoint_dir)

    if is_dist():
        dist.barrier()

    model.eval()

    pbar = tqdm(dataloader, disable=not ddp_info.is_main_process)

    with torch.no_grad(), torch.autocast(
        enabled=bool(config.training.use_amp),
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        for batch in pbar:
            batch = {
                k: (v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            result = model(batch, compute_pc_image=True)

            if config.inference.get("render_video", False):
                result = model.module.render_video(
                    result, **config.inference.render_video_config
                )

            # --- Locate prediction/GT ---
            export_results(
                result,
                config.inference.inference_out_dir,
                compute_metrics=config.inference.compute_metrics,
            )
        torch.cuda.empty_cache()

    if is_dist():
        dist.barrier()
    if ddp_info.is_main_process:
        out_dir = config.inference.inference_out_dir
        summarize_evaluation(out_dir)
        if config.inference.get("generate_website", True):
            os.system(f"python generate_html.py {out_dir}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
