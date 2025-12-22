import importlib
import os
import time
import torch
import hydra
import wandb
import gc

from easydict import EasyDict as edict
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from utils.training_utils import compute_gns
import einx


from setup import init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import (
    create_optimizer,
    create_lr_scheduler,
    auto_resume_job,
    print_rank0,
)
from utils.logging_utils import get_gpu_stats, make_profiler, cuda_timer

amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}
# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


@hydra.main(version_base=None, config_path="./configs")
def main(config):
    # Load config and read(override) arguments from CLI
    # config = init_config()
    config = OmegaConf.to_container(config, resolve=True)
    print(config)
    config = edict(config)
    config.pop("machine")

    # Set up wandb and backup source code
    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    # Load dataset
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)
    batch_size_per_gpu = config.training.batch_size_per_gpu

    val_dataset = Dataset(config, split="val")

    datasampler = DistributedSampler(dataset)
    val_datasampler = DistributedSampler(val_dataset)
    num_workers = config.training.num_workers
    # Construct kwargs to avoid errors when num_workers=0 (prefetch_factor/persistent_workers not allowed)
    dl_kwargs = dict(
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = config.training.prefetch_factor
    dataloader = DataLoader(dataset=dataset, sampler=datasampler, **dl_kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, sampler=val_datasampler, **dl_kwargs)  # type: ignore

    total_train_steps = config.training.train_steps
    grad_accum_steps = config.training.grad_accum_steps
    total_param_update_steps = total_train_steps
    total_train_steps = (
        total_train_steps * grad_accum_steps
    )  # real train steps when using gradient accumulation
    total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
    total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    if config.profiling.enable_channels_last:
        print("[bold green]Converting model to channels last...[/bold green]")
        model = model.to(memory_format=torch.channels_last)
    if config.profiling.enable_compile:
        print("[bold green]Compiling model...[/bold green]")
        model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_info.local_rank])

    optimizer, optimized_param_dict, all_param_dict = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    optim_param_list = list(optimized_param_dict.values())

    scheduler_type = config.training.get("scheduler_type", "cosine")
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_param_update_steps,
        config.training.warmup,
        scheduler_type=scheduler_type,
    )

    if config.training.get("resume_ckpt", "") != "":
        ckpt_load_path = config.training.resume_ckpt
    else:
        ckpt_load_path = config.training.checkpoint_dir
    reset_training_state = config.training.get("reset_training_state", False)
    optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
        ckpt_load_path,
        model,
        optimizer,
        lr_scheduler,
        reset_training_state,
    )

    if hasattr(dataset, "set_global_step"):
        dataset.set_global_step(cur_param_update_step)
    if hasattr(val_dataset, "set_global_step"):
        warmup_threshold = getattr(val_dataset, "data_warmup_length", 0)
        val_dataset.set_global_step(max(cur_param_update_step, warmup_threshold))

    dataloader_iter = iter(dataloader)
    val_dataloader_iter = iter(val_dataloader)

    enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=enable_grad_scaler)
    print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
    dist.barrier()

    start_train_step = cur_train_step
    model.train()

    last_epoch = -1  # for DistributedSampler.set_epoch at epoch boundary
    with make_profiler(config.profiling) as prof:
        while cur_train_step <= total_train_steps:
            tic = time.time()
            cur_epoch = int(
                cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset)
            )
            if cur_epoch != last_epoch:
                datasampler.set_epoch(cur_epoch)
                dataloader_iter = iter(dataloader)
                last_epoch = cur_epoch
            try:
                if hasattr(dataset, "set_global_step"):
                    dataset.set_global_step(cur_param_update_step)
                with cuda_timer(config.profiling.enable_cuda_timers, "dataloader"):
                    data = next(dataloader_iter)
            except StopIteration:
                print(
                    f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while..."
                )
                datasampler.set_epoch(cur_epoch)
                dataloader_iter = iter(dataloader)
                if hasattr(dataset, "set_global_step"):
                    dataset.set_global_step(cur_param_update_step)
                data = next(dataloader_iter)

            # pinned memory + non_blocking=True 可减少主机→GPU 拷贝等待
            batch = {
                k: (
                    v.to(ddp_info.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in data.items()
            }


            with cuda_timer(config.profiling.enable_cuda_timers, "forward_pass"):
                with torch.autocast(
                    enabled=config.training.use_amp,
                    device_type="cuda",
                    dtype=amp_dtype_mapping[config.training.amp_dtype],
                ):
                    ret_dict = model(batch)
            
            if cur_train_step % config.training.gns_every == 0 and config.training.compute_gns:
                per_example_loss = einx.mean("b v 3 h w -> b", (ret_dict.target.image - ret_dict.render) ** 2)
                gns_value = compute_gns(per_example_loss, model)
                if ddp_info.is_main_process:
                    print(f"GNS value: {gns_value} at step {cur_train_step}")
                    wandb.log({"gns_value": gns_value}, step=cur_train_step)

            if cur_train_step % config.training.get("val_every", 2000) == 0:
                if ddp_info.is_main_process:
                    print(f"Validating at step {cur_train_step}")
                if hasattr(val_dataset, "set_global_step"):
                    warmup_threshold = getattr(val_dataset, "data_warmup_length", 0)
                    val_dataset.set_global_step(max(cur_param_update_step, warmup_threshold))
                val_dict = do_validation(
                    model,
                    val_dataloader_iter,
                    val_datasampler,
                    val_dataloader,
                    config,
                    ddp_info,
                    cur_epoch,
                    cur_train_step,
                )
                model.train()
                if ddp_info.is_main_process:
                    loss_dict = val_dict.loss_metrics
                    loss_dict = {"val/" + k: v for k, v in loss_dict.items()}
                    wandb.log(loss_dict, step=cur_train_step)

            update_grads = (
                cur_train_step + 1
            ) % grad_accum_steps == 0 or cur_train_step == total_train_steps
            with cuda_timer(config.profiling.enable_cuda_timers, "backward_pass"):
                if update_grads:
                    scaler.scale(
                        ret_dict.loss_metrics.loss / grad_accum_steps
                    ).backward()
                else:
                    with model.no_sync():  # no sync grads for efficiency
                        scaler.scale(
                            ret_dict.loss_metrics.loss / grad_accum_steps
                        ).backward()
            cur_train_step += 1

            export_inter_results = ((cur_train_step - 1) == start_train_step) or (
                cur_train_step % config.training.vis_every == 0
            )

            skip_optimizer_step = False
            # Skip optimizer step if loss is NaN or Inf
            if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(
                ret_dict.loss_metrics.loss
            ):
                print(f"NaN or Inf loss detected, skip this iteration")
                skip_optimizer_step = True
                ret_dict.loss_metrics.loss.data = torch.zeros_like(
                    ret_dict.loss_metrics.loss
                )

            total_grad_norm = None
            # Check gradient norm and update optimizer if everything is fine
            if update_grads and (not skip_optimizer_step):
                # Unscales the gradients
                scaler.unscale_(optimizer)
                # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
                with torch.no_grad():
                    for n, p in optimized_param_dict.items():
                        if p.requires_grad and (p.grad is not None):
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

                # visualize the grad norm of each layer of our transformer
                if ddp_info.is_main_process and config.training.get(
                    "log_grad_norm_details", False
                ):
                    grad_norms = {}  # Dictionary to store norms per layer
                    for name, param in model.named_parameters():
                        if (
                            param.grad is not None
                        ):  # Some parameters might not have gradients
                            grad_norms[name] = (
                                param.grad.detach().norm().item()
                            )  # Detach for safety
                    for layer_name, grad_norm in grad_norms.items():
                        wandb.log(
                            {"grad_norm_details/" + layer_name: grad_norm},
                            step=cur_train_step,
                        )

                total_grad_norm = 0.0
                if config.training.grad_clip_norm > 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        optim_param_list, max_norm=config.training.grad_clip_norm
                    ).item()

                    if total_grad_norm > config.training.grad_clip_norm * 2.0:
                        print(
                            f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}"
                        )

                    allowed_gradnorm = (
                        config.training.grad_clip_norm
                        * config.training.get("allowed_gradnorm_factor", 5)
                    )
                    if total_grad_norm > allowed_gradnorm:
                        skip_optimizer_step = True
                        print(
                            f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step"
                        )

                    # show grad norm in wandb if it's too large
                    display_grad_norm = (
                        total_grad_norm > config.training.grad_clip_norm * 2.0
                        or total_grad_norm > allowed_gradnorm
                    )
                    if display_grad_norm and ddp_info.is_main_process:
                        wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

                if not skip_optimizer_step:
                    with cuda_timer(
                        config.profiling.enable_cuda_timers, "optimizer_step"
                    ):
                        scaler.step(optimizer)
                        scaler.update()
                    cur_param_update_step += 1
                    if hasattr(dataset, "set_global_step"):
                        dataset.set_global_step(cur_param_update_step)

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # log and save checkpoint
            if ddp_info.is_main_process:
                loss_dict = {
                    k: float(f"{v.item():.6f}")
                    for k, v in ret_dict.loss_metrics.items()
                }
                # print in console
                if (cur_train_step % config.training.print_every == 0) or (
                    cur_train_step < 100 + start_train_step
                ):
                    gpu_stats = get_gpu_stats(torch.cuda.current_device())
                    print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
                    print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f} | GPU: {gpu_stats}\n"
                    # Add loss values
                    for k, v in loss_dict.items():
                        print_str += f"{k}: {v:.6f} | "
                    print(print_str)

                # log in wandb
                if (cur_train_step % config.training.wandb_log_every == 0) or (
                    cur_train_step < 200 + start_train_step
                ):
                    log_dict = {
                        "iter": cur_train_step,
                        "forward_pass_step": cur_train_step,
                        "param_update_step": cur_param_update_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "iter_time": time.time() - tic,
                        "grad_norm": total_grad_norm,
                        "epoch": cur_epoch,
                    }
                    log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
                    wandb.log(
                        log_dict,
                        step=cur_train_step,
                    )

                # save checkpoint
                if (cur_train_step % config.training.checkpoint_every == 0) or (
                    cur_train_step == total_train_steps
                ):
                    if isinstance(model, DDP):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()
                    checkpoint = {
                        "model": model_weights,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "fwdbwd_pass_step": cur_train_step,
                        "param_update_step": cur_param_update_step,
                    }
                    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
                    ckpt_path = os.path.join(
                        config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt"
                    )
                    torch.save(checkpoint, ckpt_path)
                    print(
                        f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}"
                    )

                # export intermediate visualization results
                if export_inter_results:
                    vis_path = os.path.join(
                        config.training.checkpoint_dir,
                        "train",
                        f"iter_{cur_train_step:08d}",
                    )
                    print(f"Exporting intermediate visualization results to {vis_path}")
                    os.makedirs(vis_path, exist_ok=True)
                    visualize_intermediate_results(vis_path, ret_dict)
                    # torch.cuda.empty_cache()
                    model.train()

            if export_inter_results:
                # torch.cuda.empty_cache()
                dist.barrier()
            if hasattr(prof, "step"):
                prof.step()

    dist.barrier()
    dist.destroy_process_group()


@torch.no_grad()
def do_validation(
    model,
    val_dataloader_iter,
    val_datasampler,
    val_dataloader,
    config,
    ddp_info,
    cur_epoch,
    cur_train_step,
):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for i in range(config.training.get("val_steps", 20)):
        try:
            with cuda_timer(config.profiling.enable_cuda_timers, "dataloader"):
                data = next(val_dataloader_iter)
        except StopIteration:
            print(
                f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while..."
            )
            val_datasampler.set_epoch(cur_epoch)
            val_dataloader_iter = iter(val_dataloader)
            data = next(val_dataloader_iter)
            gc.collect(); torch.cuda.empty_cache()
        batch = {
            k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            result = model(batch, has_target_image=True)
    dist.barrier()
    merged_loss = {}
    for k, v in result.loss_metrics.items():
        t = (
            v
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=ddp_info.device, dtype=torch.float32)
        )
        t = t.detach().to(ddp_info.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / ddp_info.world_size
        merged_loss[k] = float(f"{t.item():.6f}")
    result.loss_metrics = merged_loss
    if ddp_info.is_main_process:
        vis_path = os.path.join(
            config.training.checkpoint_dir, "val", f"iter_{cur_train_step:08d}"
        )
        print(f"Exporting intermediate visualization results to {vis_path}")
        os.makedirs(vis_path, exist_ok=True)
        visualize_intermediate_results(vis_path, result)

    return result


if __name__ == "__main__":
    main()
