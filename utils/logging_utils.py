import torch
import pynvml
import contextlib, torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats(device_id=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    used_GB = meminfo.used / 1024**3
    total_GB = meminfo.total / 1024**3
    return f"{used_GB:.2f} / {total_GB:.2f} GB | {util.gpu}%"




@contextlib.contextmanager
def cuda_timer(enabled: bool, name: str):
    if not enabled:
        yield; return
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); start.record()
    try:
        yield
    finally:
        end.record(); torch.cuda.synchronize()
        print(f"[TIMER] {name}: {start.elapsed_time(end):.3f} ms")

def make_profiler(cfg):
    if not cfg.enable_profiler:
        return contextlib.nullcontext()
    sch = schedule(wait=cfg.profiler.wait, warmup=cfg.profiler.warmup,
                   active=cfg.profiler.active, repeat=cfg.profiler.repeat)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sch,
        on_trace_ready=tensorboard_trace_handler(cfg.profiler.logdir),
        record_shapes=True, profile_memory=True, with_stack=True, with_modules=True
    )
