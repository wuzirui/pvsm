# -*- coding: utf-8 -*-
import os
import json
import random
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

try:
    import imageio.v3 as iio
except Exception:
    iio = None

OVERLAP_PRESET = {
    "small": (0.05, 0.3),
    "medium": (0.3, 0.55),
    "large": (0.55, 0.8),
}

# ---------------- depth readers ----------------


def _read_exr_depth(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None and iio is not None:
        arr = iio.imread(str(path))
    if arr is None:
        raise RuntimeError(f"Cannot read EXR: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        chans = [arr[..., i] for i in range(arr.shape[-1])]
        finite_cnt = [np.isfinite(c).sum() for c in chans]
        best = int(np.argmax(finite_cnt)) if finite_cnt else 0
        return chans[best].astype(np.float32)
    raise ValueError(f"Unexpected EXR shape {arr.shape} for {path}")


def _read_depth_any(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        return None
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(path).astype(np.float32)
        elif ext == ".exr":
            arr = _read_exr_depth(path)
        else:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            arr = arr.astype(np.float32)
    except Exception:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    return arr


# ---------------- main dataset ----------------


class WaiBenchmarkDataset(Dataset):
    """
    Benchmark loader for generated groups (roll/yawpitch/fov/anisopixel/scale_meta).

    Config fields expected:
      config.benchmark.BENCH_DIR: str  # <out_root>，contains five group subdirectories (roll/yawpitch/fov/anisopixel/scale_meta)
      config.benchmark.group: str      # 'roll'|'yawpitch'|'fov'|'anisopixel'|'scale_meta'
      config.benchmark.split: str      # 'test'|'val'... only used for identification/logging
      config.benchmark.eval_use_index_file: bool = False
      config.benchmark.index_list.path: str
      config.benchmark.index_list.overlap_min / overlap_max: float
      config.benchmark.index_list.overlap_preset: 'small'|'medium'|'large'
      config.benchmark.fixed_length: Optional[int] = None
      config.benchmark.overfitting_mode: bool = False
      config.benchmark.allow_resample_on_error: bool = True
      config.benchmark.max_resample_attempts: int = 16
      config.benchmark.load_depth: bool = True
      config.benchmark.load_mask: bool = True

    Shared/training related:
      config.training.num_views: int
      config.training.square_crop: bool = False
      config.model.image_tokenizer.image_size: Optional[int]
      config.model.image_tokenizer.patch_size: int = 14

    Returns dict:
      {
        'image': [V,3,H,W], 'c2w': [V,4,4], 'fxfycxcy': [V,4],
        'index': [V,2], 'scene_name': str,
        'depth': [V,1,H,W],        # if available
        'valid_mask': [V,1,H,W],   # if available
        'transforms': list[dict],  # if available
      }
    """

    def __init__(self, config, split: str = "test"):
        super().__init__()
        self.config = config
        self.split = split  # used for identification/logging

        # ---- resolve cfg namespace ----
        bench = getattr(config, "benchmark", None) or getattr(config, "wai_bench", None)
        assert (
            bench is not None
        ), "Please set config.benchmark.* (or config.wai_bench.*)"

        self.BENCH_DIR = Path(bench.BENCH_DIR)
        self.group = str(getattr(bench, "group", "roll"))
        self.num_views = int(config.training.num_views)

        self.overfitting_mode = bool(getattr(bench, "overfitting_mode", False))
        self.allow_resample_on_error = bool(
            getattr(bench, "allow_resample_on_error", True)
        )
        self.max_resample_attempts = int(getattr(bench, "max_resample_attempts", 16))
        self.fixed_length = None

        self.load_depth = bool(getattr(bench, "load_depth", True))
        self.load_mask = bool(getattr(bench, "load_mask", True))

        self.square_crop = bool(getattr(config.training, "square_crop", False))
        self.img_size = getattr(config.model.image_tokenizer, "image_size", None)
        self.patch = int(getattr(config.model.image_tokenizer, "patch_size", 14))

        # ---- sequence list ----
        bench_group_root = self.BENCH_DIR / self.group
        assert bench_group_root.exists(), f"Group root not found: {bench_group_root}"

        use_index = bool(getattr(bench, "eval_use_index_file", False))
        self.sequence_list: List[str] = []

        if use_index:
            with open(bench.index_list.path, "r") as f:
                idx_map = json.load(f)
            # only keep scenes in this group
            for seq_name, seq_info in idx_map.items():
                if seq_info is None or "overlap" not in seq_info:
                    continue
                # if (bench_group_root / seq_name / "scene_meta.json").exists():
                self.sequence_list.append(seq_name)
            print(
                f"[WaiBenchmarkDataset] group='{self.group}' use_index=True kept={len(self.sequence_list)}"
            )
        else:
            print(f"Scanning group directory: {bench_group_root}")
            # scan group directory
            for d in sorted(bench_group_root.iterdir()):
                if not d.is_dir():
                    continue
                if (d / "scene_meta.json").exists():
                    self.sequence_list.append(d.name)
            print(
                f"[WaiBenchmarkDataset] group='{self.group}' scanned kept={len(self.sequence_list)}"
            )

    # ---------------- basic utils ----------------

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return 10000 if self.overfitting_mode else len(self.sequence_list)

    def _load_scene_frames(self, scene_dir: Path) -> Tuple[List[Dict], int, int]:
        """Load meta frames; resolve paths; return (frames_resolved, W, H)."""
        meta_p = scene_dir / "scene_meta.json"
        with meta_p.open("r") as f:
            meta = json.load(f)
        frames = meta.get("frames", [])
        W = int(meta.get("w", 0))
        H = int(meta.get("h", 0))

        recs: List[Dict] = []
        for fr in frames:
            img_rel = fr.get("image") or fr.get("file_path")
            assert img_rel is not None, f"Missing image in {scene_dir}"
            img_path = scene_dir / img_rel

            fx = float(fr["fl_x"])
            fy = float(fr["fl_y"])
            cx = float(fr["cx"])
            cy = float(fr["cy"])
            c2w = np.array(fr["transform_matrix"], dtype=np.float32).reshape(4, 4)

            depth_rel = fr.get("depth")
            depth_path = (scene_dir / depth_rel) if depth_rel else None

            mask_rel = fr.get("valid_mask")
            mask_path = (scene_dir / mask_rel) if mask_rel else None

            recs.append(
                dict(
                    img_path=img_path,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                    W=W,
                    H=H,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    transform=fr.get("transform", None),
                )
            )
        return recs, W, H

    # ---------------- image/depth preprocessing ----------------

    def _read_img(self, p: Path) -> Image.Image:
        return Image.open(p).convert("RGB")

    def _resize_and_crop(
        self, img: Image.Image, depth: Optional[np.ndarray], mask: Optional[np.ndarray]
    ):
        orig_w, orig_h = img.size
        if self.img_size and self.img_size > 0:
            new_h = self.img_size
            new_w = int(round(new_h / orig_h * orig_w / self.patch) * self.patch)
            new_w = max(new_w, self.patch)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            if depth is not None:
                depth = cv2.resize(
                    depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
            if mask is not None:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = (mask > 0).astype(np.uint8)
        else:
            new_w, new_h = orig_w, orig_h

        sx_resize = new_w / float(orig_w)
        sy_resize = new_h / float(orig_h)

        sw = sh = 0.0
        if self.square_crop:
            min_size = min(new_w, new_h)
            sw = (new_w - min_size) * 0.5
            sh = (new_h - min_size) * 0.5
            sw_i, sh_i = int(sw), int(sh)
            img = img.crop((sw_i, sh_i, sw_i + min_size, sh_i + min_size))
            if depth is not None:
                depth = depth[sh_i : sh_i + min_size, sw_i : sw_i + min_size]
            if mask is not None:
                mask = mask[sh_i : sh_i + min_size, sw_i : sw_i + min_size]

            new_w, new_h = min_size, min_size

        return img, depth, mask, (sx_resize, sy_resize, sw, sh, new_w, new_h)

    @staticmethod
    def _adjust_intrinsics(fx, fy, cx, cy, sx, sy, sw, sh):
        fx = fx * sx
        fy = fy * sy
        cx = cx * sx - float(sw)
        cy = cy * sy - float(sh)
        return fx, fy, cx, cy

    @staticmethod
    def preprocess_poses(c2ws: torch.Tensor) -> torch.Tensor:
        centers = c2ws[:, :3, 3].mean(0)
        fwd = F.normalize(c2ws[:, :3, 2].mean(0), dim=0)
        down = c2ws[:, :3, 1].mean(0)
        right = F.normalize(torch.cross(down, fwd, dim=0), dim=0)
        down = F.normalize(torch.cross(fwd, right, dim=0), dim=0)
        avg = torch.eye(4, device=c2ws.device)
        avg[:3, :3] = torch.stack([right, down, fwd], dim=-1)
        avg[:3, 3] = centers
        return torch.linalg.inv(avg) @ c2ws

    # ---------------- Dataset API ----------------

    def __getitem__(self, idx: int, attempts: int = 0):
        if self.overfitting_mode:
            idx = 0
        if self.fixed_length is not None:
            idx = random.randint(0, len(self.sequence_list) - 1)
        if attempts >= self.max_resample_attempts:
            raise Exception("Failed to sample a valid sequence after multiple retries.")
        try:
            data = self.get_data(idx)
        except Exception as e:
            if self.allow_resample_on_error:
                print(f"[WaiBenchmarkDataset] Error seq={self.sequence_list[idx]}: {e}")
                with open("wai_benchmark_error.log", "a") as f:
                    f.write(
                        f"Error loading data for seq={self.sequence_list[idx]}: {e}\n"
                    )
                    f.write(traceback.format_exc())
                return self.__getitem__(random.randint(0, len(self) - 1), attempts + 1)
            else:
                raise e
        return data

    def get_data(self, idx: int):
        scene_name = self.sequence_list[idx % len(self.sequence_list)]
        scene_dir = self.BENCH_DIR / self.group / scene_name
        recs, base_W, base_H = self._load_scene_frames(scene_dir)

        # directly use the frames order in the benchmark set (already context... + target...)
        chosen = recs  # length V

        images, intrs, c2ws, depths, vmasks, tf_list = [], [], [], [], [], []

        for local_id, rec in enumerate(chosen):
            img = self._read_img(rec["img_path"])
            depth_np = _read_depth_any(rec["depth_path"]) if self.load_depth else None
            mask_np = None
            if (
                self.load_mask
                and rec.get("mask_path") is not None
                and rec["mask_path"].exists()
            ):
                mask_np = (np.array(Image.open(rec["mask_path"])) > 127).astype(
                    np.uint8
                )
            else:
                mask_np = np.ones((base_H, base_W), dtype=np.uint8)

            # resize/crop
            img, depth_np, mask_np, (sx, sy, sw, sh, new_w, new_h) = (
                self._resize_and_crop(img, depth_np, mask_np)
            )

            # to tensor
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor_img = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            images.append(tensor_img)

            # intrinsics after resize/crop
            fx, fy, cx, cy = self._adjust_intrinsics(
                rec["fx"], rec["fy"], rec["cx"], rec["cy"], sx, sy, sw, sh
            )
            intrs.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))

            # depth tensor
            if self.load_depth:
                if depth_np is None:
                    dep_t = torch.zeros(
                        1, tensor_img.shape[1], tensor_img.shape[2], dtype=torch.float32
                    )
                else:
                    dep_t = torch.from_numpy(depth_np).float().unsqueeze(0)
                depths.append(dep_t)

            # valid mask tensor
            if self.load_mask:
                if mask_np is None:
                    # if mask is missing: default all 1
                    msk = torch.ones(
                        1, tensor_img.shape[1], tensor_img.shape[2], dtype=torch.float32
                    )
                else:
                    msk = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
                vmasks.append(msk)

            c2w = rec["c2w"]
            if c2w.shape != (4, 4):
                c2w = np.array(c2w, dtype=np.float32).reshape(4, 4)
            c2ws.append(torch.from_numpy(c2w).float())

            tf_list.append(rec.get("transform", None))

        images = torch.stack(images)
        intrs = torch.stack(intrs)
        c2ws = torch.stack(c2ws)
        c2ws = self.preprocess_poses(c2ws)  # align with original loader

        ret = {
            "image": images,
            "c2w": c2ws,
            "fxfycxcy": intrs,
            "index": torch.stack(
                [
                    torch.arange(len(chosen), dtype=torch.long),
                    torch.full((len(chosen),), idx, dtype=torch.long),
                ],
                dim=1,
            ),
            "scene_name": scene_name,
        }
        if self.load_depth:
            ret["depth"] = torch.stack(depths)
        if self.load_mask:
            ret["valid_mask"] = torch.stack(vmasks)
        # ret["transforms"] = tf_list
        return ret
