# Used for MAE training

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))


def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


class PureImageDataset(Dataset):
    """
    Pure image Dataset (based on splits.json, no scanning; read configuration from cfg.data_cfg)

    Expected cfg.data_cfg fields (all have default values, can be omitted):
      - root_dir: str, data root directory (note that this version concatenates paths as root/scene/filename)
      - splits_json: str, splits.json path
      - split: str, "train" | "val" | "test"
      - num_views: int, number of frames per sample (default 4)
      - min_seq_len: int, minimum sequence length filter (default equal to num_views)
      - image_size: Optional[int], if given, scale height to this value (width is scaled proportionally)
      - square_crop: bool, whether to center square crop (default False)
      - patch_size: int, if >0, align "width" after scaling to this integer multiple (default 14)
      - min_frame_dist/max_frame_dist: view selection first and last interval (default 25/90)
      - fixed_length: Optional[int], if given then __len__ returns this value
      - overfitting_mode: bool (default False)
      - allow_resample_on_error: bool (default True)
      - max_resample_attempts: int (default 16)

    Pseudo parameter control (optional):
      - fake_intrinsics: bool (default True)
      - fake_fov_deg: float (default 60.0)
      - fake_poses: bool (default True)
      - fake_pose_mode: "identity" | "line" | "circle" (default "identity")
      - fake_baseline: float, translation scale (default 0.1)

    Return:
      {
        "image": Tensor [V,3,H,W] (float32, 0..1),
        "index": Tensor [V,2]   # (frame_idx_in_scene, scene_idx)
        "scene_name": str,
        "c2w": Tensor [V,4,4],
        "fxfycxcy": Tensor [V,4],
      }
    """

    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        dcfg = _get(cfg, "data_cfg")
        assert dcfg is not None, "cfg.data_cfg is required."

        self.root = Path(_get(dcfg, "root_dir", "."))
        self.split = split
        if self.split == "val":
            self.split = "test"  # if you don't want to map, you can delete this

        self.num_views = int(_get(dcfg, "num_views", 4))
        self.min_seq_len = int(_get(dcfg, "min_seq_len", self.num_views))
        self.image_size = _get(dcfg, "image_size", None)
        self.image_size = (
            int(self.image_size) if self.image_size not in (None, 0) else None
        )
        self.square_crop = bool(_get(dcfg, "square_crop", False))
        self.patch_size = int(_get(dcfg, "patch_size", 14))
        self.min_frame_dist = int(_get(dcfg, "min_frame_dist", 25))
        self.max_frame_dist = int(_get(dcfg, "max_frame_dist", 90))
        self.fixed_length = _get(dcfg, "fixed_length", None)
        self.fixed_length = (
            int(self.fixed_length) if self.fixed_length not in (None, 0) else None
        )
        self.overfitting_mode = bool(_get(dcfg, "overfitting_mode", False))
        self.allow_resample_on_error = bool(_get(dcfg, "allow_resample_on_error", True))
        self.max_resample_attempts = int(_get(dcfg, "max_resample_attempts", 16))

        # pseudo parameter control
        self.fake_intrinsics = bool(_get(dcfg, "fake_intrinsics", True))
        self.fake_fov_deg = float(_get(dcfg, "fake_fov_deg", 60.0))
        self.fake_poses = bool(_get(dcfg, "fake_poses", True))
        self.fake_pose_mode = str(_get(dcfg, "fake_pose_mode", "identity"))
        self.fake_baseline = float(_get(dcfg, "fake_baseline", 0.1))

        splits_json = Path(_get(dcfg, "splits_json", "splits.json"))
        data = json.loads(splits_json.read_text())
        if self.overfitting_mode:
            self.split = "train"
        if self.split not in data:
            raise KeyError(f"Split '{self.split}' not found in {splits_json}")

        split_map: Dict[str, Dict] = data[self.split]

        def _count_frames(meta: Dict) -> int:
            if isinstance(meta, dict) and "image_filenames" in meta:
                return len(meta["image_filenames"])
            return int(meta.get("num_frames", 0))

        items = [(scene, meta) for scene, meta in split_map.items()]
        kept = [(s, m) for s, m in items if _count_frames(m) >= self.min_seq_len]
        dropped = [
            (s, _count_frames(m))
            for s, m in items
            if _count_frames(m) < self.min_seq_len
        ]

        self.scene_names: List[str] = [s for s, _ in kept]
        self.scene_meta: Dict[str, Dict] = {s: m for s, m in kept}

        print(
            f"[PureImageDataset] split={self.split} total={len(items)} "
            f"kept={len(kept)} dropped={len(dropped)} (min_seq_len={self.min_seq_len})"
        )

    # ---------------- basic ----------------

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return 10000 if self.overfitting_mode else len(self.scene_names)

    def __getitem__(self, idx: int, attempts: int = 0):
        if self.overfitting_mode:
            idx = 0
        if self.fixed_length is not None:
            idx = random.randint(0, max(0, len(self.scene_names) - 1))

        if attempts >= self.max_resample_attempts:
            raise RuntimeError(
                "Exceeded max_resample_attempts while sampling a sequence."
            )

        try:
            return self._get_item_impl(idx)
        except Exception:
            if not self.allow_resample_on_error:
                raise
            return self.__getitem__(random.randint(0, len(self) - 1), attempts + 1)

    # --------------- impl ---------------

    def _get_item_impl(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_name = self.scene_names[idx % len(self.scene_names)]
        meta = self.scene_meta[scene_name]
        fnames: List[str] = meta["image_filenames"]
        n = len(fnames)
        if n < self.num_views:
            raise RuntimeError(
                f"Scene '{scene_name}' has {n} frames (< num_views={self.num_views})."
            )

        sel = self._select_views(n)

        # note: this version concatenates paths as root/scene/filename
        chosen_paths = [(self.root / scene_name / fnames[i]).resolve() for i in sel]

        # read images
        images = [self._load_one_image(p) for p in chosen_paths]
        images = torch.stack(images, dim=0)  # [V,3,H,W]

        # fake intrinsics (based on final H,W and fixed FOV)
        V, _, H, W = images.shape
        fxfycxcy = torch.tensor(
            [[1.0, 1.0, (W - 1) / 2.0, (H - 1) / 2.0]] * V, dtype=torch.float32
        )

        # fake poses
        c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(len(sel), 1, 1)

        # index: [V,2] = (frame_idx_in_scene, scene_idx)
        idx_tensor = torch.tensor(sel, dtype=torch.long).unsqueeze(-1)
        scene_tensor = torch.full_like(idx_tensor, idx, dtype=torch.long)
        index = torch.cat([idx_tensor, scene_tensor], dim=-1)

        return {
            "image": images,
            "index": index,
            "scene_name": scene_name,
            "c2w": c2w,  # [V,4,4]
            "fxfycxcy": fxfycxcy,  # [V,4]
        }

    def _select_views(self, num_frames: int) -> List[int]:
        if num_frames < self.num_views:
            return None
        min_frame_dist = self.min_frame_dist
        max_frame_dist = self.max_frame_dist
        min_d = min_frame_dist
        max_d = min(num_frames - 1, max_frame_dist)
        if max_d < min_d:
            raise Exception(
                f"Error: have total of {num_frames} frames, but max_frame_dist is {max_d} and min_frame_dist is {min_d}."
            )
        d = random.randint(min_d, max_d)
        start = random.randint(0, num_frames - d - 1)
        end = start + d
        mids = random.sample(range(start + 1, end), self.num_views - 2)
        return [start, end] + mids

    def _load_one_image(self, path: Path) -> torch.Tensor:
        """read -> optional resize -> optional center square crop -> Tensor[3,H,W] in [0,1]"""
        img = Image.open(path).convert("RGB")

        if self.square_crop:
            img = _center_square_crop(img)

        if self.image_size and self.image_size > 0:
            ow, oh = img.size
            new_h = self.image_size
            new_w = int(round(ow * (new_h / float(oh))))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        return ten

    # ---------------- pseudo parameter generation ----------------

    @staticmethod
    def _fake_intr_from_tensor(
        img_tensor: torch.Tensor, fov_deg: float
    ) -> torch.Tensor:
        """
        Generate (fx,fy,cx,cy) from image Tensor [3,H,W] and fixed FOV.
        Assume square pixels, fx/fy are calculated as W/H.
        """
        _, H, W = img_tensor.shape
        fov = np.deg2rad(float(fov_deg))
        # fx = 0.5*W / tan(FOVx/2), fy = 0.5*H / tan(FOVy/2); if FOVx=FOVy=fov then
        fx = 0.5 * W / np.tan(fov / 2.0)
        fy = 0.5 * H / np.tan(fov / 2.0)
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    @staticmethod
    def _eye(n: int) -> torch.Tensor:
        return torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)

    def _fake_poses(
        self, V: int, mode: str = "identity", baseline: float = 0.1
    ) -> torch.Tensor:
        """
        Generate V 4x4 c2w. Only ensure shape correctness, geometric meaning is meaningless, only for compatibility with downstream.
        mode:
          - identity: full identity matrix
          - line: translation along X axis with equal spacing
          - circle: planar circular translation (rotation still Identity)
        """
        c2w = self._eye(V)
        if mode == "identity":
            return c2w

        if mode == "line":
            # center-aligned equal spacing translation
            offs = torch.linspace(-(V - 1) / 2.0, (V - 1) / 2.0, V) * float(baseline)
            c2w[:, 0, 3] = offs  # x translation
            return c2w

        if mode == "circle":
            radius = max(baseline, 1e-6) * max(1.0, (V - 1) * 0.5)
            angles = torch.linspace(0, 2 * np.pi * (V - 1) / V, V)
            c2w[:, 0, 3] = radius * torch.cos(angles)  # x
            c2w[:, 2, 3] = radius * torch.sin(angles)  # z
            return c2w

        # unknown mode fallback
        return c2w
