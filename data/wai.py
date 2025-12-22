import os
import json
import traceback
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict

OVERLAP_PRESET = {
    "small": (0.05, 0.3),
    "medium": (0.3, 0.55),
    "large": (0.55, 0.8),
    "all": (0.0, 1.0),
}

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
try:
    import imageio.v3 as iio
except Exception:
    iio = None


def _load_full_list_counts(full_list_path: Path) -> List[tuple[str, int]]:
    """
    Read root/split/full_list.txt in two columns format: <scene>\t<frame_count>
    Return [(scene_name, frame_count), ...]. If second column is not an integer, treat it as 0.
    """
    if not full_list_path.exists():
        raise FileNotFoundError(f"full_list.frames.txt not found: {full_list_path}")

    pairs: List[tuple[str, int]] = []
    with full_list_path.open("r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            assert len(parts) == 2, f"Invalid line: {line}"
            name = parts[0]
            n = int(parts[1])
            pairs.append((name, n))
    return pairs


def _read_exr_depth(path: Path) -> np.ndarray:
    """Robust EXR -> float32 (H,W). Return single channel directly, select the channel with most valid pixels for multiple channels."""
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


def _create_full_list(full_list_path: Path):
    """scan the directory and create the full list"""
    from datetime import datetime
    with full_list_path.open("w") as f:
        print(f"Scanning {full_list_path.parent} for full list")
        for scene_dir in full_list_path.parent.iterdir():
            if scene_dir.is_dir():
                print(f"Adding {scene_dir.name} to full list")
                f.write(f"{scene_dir.name}\t{datetime.now().isoformat()}\n")


def _load_full_list(full_list_path: Path) -> List[str]:
    """root/split/full_list.txt in two columns format: <scene>\t<iso_time>, only take the first column."""
    if not full_list_path.exists():
        print(f"full_list.txt not found: {full_list_path}, creating...")
        _create_full_list(full_list_path)

    names = []
    with full_list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.append(line.split()[0])
    return names


class WaiDataset(Dataset):
    """
    Lazy WAI Dataset:
      - Don't pre-read all frames of all scenes in __init__;
      - Read scene_meta.json on demand for each scene in __getitem__
      - Build the required frames for the current sample, then load and preprocess the images/depths.
    Return:
      {
        'image': [V,3,H,W], 'c2w': [V,4,4], 'fxfycxcy': [V,4],
        'index': [V,2], 'scene_name': str,
        # optional:
        'depth': [V,1,H,W], 'depth_mask': [V,1,H,W]
      }
    """

    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.ROOT = Path(config.wai.WAI_DIR)
        self.split = "test" if split == "val" else split
        self.num_views = int(config.training.num_views)
        self.min_num_images = int(config.wai.min_num_images)
        self.square_crop = bool(config.training.get('square_crop', False))
        self.overfitting_mode = bool(config.wai.overfitting_mode)
        self.allow_resample_on_error = bool(config.wai.get('allow_resample_on_error', True))
        self.max_resample_attempts = int(config.wai.get('max_resample_attempts', 16))
        if self.overfitting_mode:
            self.split = "train"
        self.load_depth = bool(config.wai.get('load_depth', True))
        self.load_mask = bool(config.wai.get('load_mask', True))

        self.img_size = getattr(config.model.image_tokenizer, 'image_size', None)
        self.patch = getattr(config.model.image_tokenizer, 'patch_size', 14)

        self.min_frame_dist = config.training.view_selector.get('min_frame_dist', 45)
        self.max_frame_dist = config.training.view_selector.get('max_frame_dist', 90)
        assert self.max_frame_dist >= self.min_frame_dist, f"max_frame_dist={self.max_frame_dist} must be >= min_frame_dist={self.min_frame_dist}"
        assert self.min_frame_dist >= self.num_views, f"min_frame_dist={self.min_frame_dist} must be >= num_views={self.num_views}"
        self.data_warmup_length = int(config.training.view_selector.get('data_warmup_length', 0))
        self.warmup_min_frame_dist = int(config.training.view_selector.get('warmup_min_frame_dist', 25))
        self.warmup_max_frame_dist = int(config.training.view_selector.get('warmup_max_frame_dist', 25))

        initial_step = 0 if self.split == "train" else self.data_warmup_length
        self._global_step = mp.Value("i", int(initial_step))

        if config.wai.get('fixed_length', None):
            """ set dataset length to a fixed number to avoid frequent StopIteration errors in overfitting mode """
            self.fixed_length = int(config.wai.fixed_length)
        else:
            self.fixed_length = None

        if config.wai.get('eval_use_index_file', False) and self.split == "test":
            """ use index file to load sequences, for reproducable evaluation on different overlap settings """
            self.use_index_file = True
            with open(config.wai.index_list.path, 'r') as f:
                self.index_list = json.load(f)
            overlap_min = float(config.wai.index_list.get('overlap_min', 0.0))
            overlap_max = float(config.wai.index_list.get('overlap_max', 1.0))
            if config.wai.index_list.get('overlap_preset', None) is not None:
                overlap_min, overlap_max = OVERLAP_PRESET[config.wai.index_list.get('overlap_preset')]
            self.sequence_list = []
            skip = 0
            for seq_name, seq_info in self.index_list.items():
                if seq_info is None or "overlap" not in seq_info:
                    skip += 1
                    continue
                if overlap_min <= seq_info["overlap"] <= overlap_max:
                    self.sequence_list.append(seq_name)
            print(f"Loaded {len(self.sequence_list)} sequences from index file {config.wai.index_list.path}, skipped {skip} sequences")
        else:
            self.use_index_file = False

        if not self.use_index_file:
            # scan the directory under <ROOT>/<split>, filter by meta['frames'] length
            min_seq_len = int(getattr(self.config.wai, "min_seq_len", self.min_num_images))
            # use the new full_list (second column is frame count) to filter by shortest seqlen
            pairs = _load_full_list_counts(self.ROOT / self.split / "full_list.frames.txt")
            kept = [s for s, n in pairs if n >= min_seq_len]
            dropped = [(s, n) for s, n in pairs if n < min_seq_len]
            self.sequence_list = kept

            print(
                f"[WaiDataset] Split='{self.split}' | total={len(pairs)} "
                f"kept={len(kept)} dropped={len(dropped)} (min_seq_len={min_seq_len})"
            )

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return 10000 if self.overfitting_mode else len(self.sequence_list)

    # ---------------------- per-scene lazy metadata ----------------------

    def _load_scene_records(self, scene_name: str) -> Optional[List[Dict]]:
        """Read scene_meta.json for a single scene, and convert it to the list of records we use."""
        scene_dir = self.ROOT / self.split / scene_name
        meta_p = scene_dir / "scene_meta.json"
        assert meta_p.exists(), f"scene_meta.json not found: {meta_p}"

        with meta_p.open("r") as f:
            meta = json.load(f)

        frames = meta.get('frames', [])
        if len(frames) < self.min_num_images:
            raise Exception(f"Not enough frames in scene {scene_name}")

        shared_K = None
        base_W = base_H = None
        if meta.get('shared_intrinsics', False):
            fx = float(meta['fl_x']); fy = float(meta['fl_y'])
            cx = float(meta['cx']);  cy = float(meta['cy'])
            shared_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            base_W = int(meta['w']); base_H = int(meta['h'])

        recs = []
        for fr in frames:
            img_rel = fr.get('image') or fr.get('file_path')
            if not img_rel:
                continue
            img_path = scene_dir / img_rel

            c2w = np.array(fr['transform_matrix'], dtype=np.float32).reshape(4, 4)

            if all(k in fr for k in ('fl_x', 'fl_y', 'cx', 'cy')):
                fx = float(fr['fl_x']); fy = float(fr['fl_y'])
                cx = float(fr['cx']);  cy = float(fr['cy'])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                W = int(fr.get('w', 0)); H = int(fr.get('h', 0))
                if W == 0 or H == 0:
                    with Image.open(img_path) as im0:
                        W, H = im0.size
            else:
                K = shared_K
                if K is None:
                    with Image.open(img_path) as im0:
                        W, H = im0.size
                    # no intrinsics, skip
                    continue
                else:
                    W, H = base_W, base_H

            recs.append({
                "img_path": img_path,
                "K": K, "W": W, "H": H,
                "c2w": c2w,
                "depth_rel": fr.get("depth"),
                "conf_rel": fr.get("depth_confidence"),
                "mask_rel": fr.get("mask"),
                "scene_dir": scene_dir,
            })
        return recs if len(recs) >= self.min_num_images else None

    # -------------------------- helpers --------------------------

    def set_global_step(self, step: int):
        step = int(step)
        with self._global_step.get_lock():
            if self._global_step.value != step:
                self._global_step.value = step

    def get_global_step(self) -> int:
        with self._global_step.get_lock():
            return int(self._global_step.value)

    def view_selector(self, num_frames):
        if num_frames < self.num_views:
            return None
        current_step = self.get_global_step()
        is_warmup = self.data_warmup_length > 0 and current_step < self.data_warmup_length
        min_frame_dist = self.warmup_min_frame_dist if is_warmup else self.min_frame_dist
        max_frame_dist = self.warmup_max_frame_dist if is_warmup else self.max_frame_dist
        min_d = int(min_frame_dist)
        max_d = min(num_frames - 1, int(max_frame_dist))
        if max_d < min_d:
            raise Exception(f"Error: have total of {num_frames} frames, but max_frame_dist is {max_d} and min_frame_dist is {min_d}.")
        d = random.randint(min_d, max_d)
        start = random.randint(0, num_frames - d - 1)
        end = start + d
        mids = random.sample(range(start + 1, end), self.num_views - 2)
        return [start, end] + mids

    def _read_depth_any(self, scene_dir: Path, rel_path: Optional[str]) -> Optional[np.ndarray]:
        if not self.load_depth or rel_path is None:
            return None
        p = scene_dir / rel_path
        if not p.exists():
            return None
        ext = p.suffix.lower()
        if ext == ".npy":
            d = np.load(p).astype(np.float32)
        elif ext == ".exr":
            d = _read_exr_depth(p)
        else:
            try:
                d = np.array(Image.open(p)).astype(np.float32)
            except Exception:
                d = None
        return d

    def preprocess_frames(self, records: List[Dict]):
        images, intrinsics, depths, depth_masks = [], [], [], []
        have_depth = self.load_depth
        have_mask = self.load_mask

        img_size = self.img_size
        patch = self.patch if self.patch and self.patch > 0 else 1

        for rec in records:
            img = Image.open(rec['img_path']).convert("RGB")
            orig_w, orig_h = img.size

            # read/clean depth & optional mask
            depth_np = self._read_depth_any(rec['scene_dir'], rec.get('depth_rel'))
            if depth_np is not None:
                depth_np = np.asarray(depth_np, dtype=np.float32)
                # —— robust cleaning: NaN/Inf→0, negative values truncated to 0, remove extreme outliers (optional)
                depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
                depth_np[depth_np < 0] = 0.0
            mask_np = None
            if have_mask and rec.get('mask_rel') is not None:
                mask_p = rec['scene_dir'] / rec['mask_rel']
                if mask_p.exists():
                    mask_np = np.array(Image.open(mask_p)).astype(np.uint8)
                    if mask_np.ndim == 3: mask_np = mask_np[..., 0]
                    mask_np = (mask_np > 127).astype(np.float32)

            # --- resize to (new_w, new_h) ---
            if img_size and img_size > 0:
                new_h = img_size
                new_w = int(round(new_h / orig_h * orig_w / patch) * patch)
                new_w = max(new_w, patch)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                if depth_np is not None:
                    depth_np = cv2.resize(depth_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if mask_np is not None:
                    mask_np = cv2.resize(mask_np.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    mask_np = (mask_np > 0).astype(np.float32)
            else:
                new_w, new_h = orig_w, orig_h

            # —— calculate the scaling factor "only due to resize" (before cropping) ——
            sx_resize = new_w / float(orig_w)
            sy_resize = new_h / float(orig_h)

            # --- optional center square crop (only affects principal point translation, does not affect fx/fy scaling) ---
            sw = sh = 0.0  # cropping offset (in resized pixels)
            if self.square_crop:
                min_size = min(new_h, new_w)
                sw = (new_w - min_size) * 0.5
                sh = (new_h - min_size) * 0.5
                # actually crop to a square
                sw_i, sh_i = int(sw), int(sh)
                img = img.crop((sw_i, sh_i, sw_i + min_size, sh_i + min_size))
                if depth_np is not None:
                    depth_np = depth_np[sh_i:sh_i + min_size, sw_i:sw_i + min_size]
                if mask_np is not None:
                    mask_np = mask_np[sh_i:sh_i + min_size, sw_i:sw_i + min_size]

            # image -> [3,H,W], 0..1
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor_img = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            images.append(tensor_img)

            # --- correct intrinsics transformation: scale first, then translate ---
            fx, fy, cx, cy = rec['K'][0,0], rec['K'][1,1], rec['K'][0,2], rec['K'][1,2]
            fx = fx * sx_resize
            fy = fy * sy_resize
            cx = cx * sx_resize - float(sw)
            cy = cy * sy_resize - float(sh)
            intrinsics.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))

            # depth / mask tensor
            if have_depth:
                if depth_np is None:
                    dep_t = torch.zeros(1, tensor_img.shape[1], tensor_img.shape[2], dtype=torch.float32)
                    msk_t = torch.zeros_like(dep_t)  # all 0, explicitly tell downstream "this frame has no depth"
                else:
                    dep_t = torch.from_numpy(depth_np).float().unsqueeze(0)
                    if mask_np is not None:
                        msk_t = torch.from_numpy(mask_np).float().unsqueeze(0)
                    else:
                        msk_t = (dep_t > 1e-6).float()
                depths.append(dep_t)
                depth_masks.append(msk_t)

        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics)
        if have_depth:
            depths = torch.stack(depths)
            depth_masks = torch.stack(depth_masks)

        # c2w stack same as original logic...
        c2ws = []
        for rec in records:
            c2w = rec['c2w']
            if c2w.shape != (4,4):
                c2w = np.array(c2w, dtype=np.float32).reshape(4,4)
            c2ws.append(c2w)
        c2ws = torch.from_numpy(np.stack(c2ws)).float()

        return images, intrinsics, c2ws, (depths if have_depth else None), (depth_masks if have_depth else None)

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

    # ------------------------------ main ------------------------------

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
                print(f"Error loading data for seq={self.sequence_list[idx]}: {e}...")
                print("See wai_error.log for details...")
                # append stack trace to local log
                with open("wai_error.log", "a") as f:
                    f.write(
                        f"Error loading data for seq={self.sequence_list[idx]}: {e}\n"
                    )
                    f.write(traceback.format_exc())
                return self.__getitem__(random.randint(0, len(self) - 1), attempts + 1)
            else:
                raise e
        return data

    def get_data(self, idx):
        if isinstance(idx, str):
            seq = idx
        else:
            seq = self.sequence_list[idx % len(self.sequence_list)]
        recs = self._load_scene_records(seq)
        if recs is None or len(recs) < self.min_num_images:
            raise Exception(f"Not enough frames in scene {seq}")

        if self.use_index_file:
            sel = self.index_list[seq]["context"] + self.index_list[seq]["target"]
        else:
            sel = self.view_selector(len(recs))
        if sel is None:
            raise Exception("Failed to sample a valid sequence.")

        chosen = [recs[i] for i in sel]
        imgs, intrs, c2ws, depths, depth_masks = self.preprocess_frames(chosen)
        c2ws = self.preprocess_poses(c2ws)

        idx_tensor = torch.tensor(sel, dtype=torch.long).unsqueeze(-1)
        scene_tensor = torch.full_like(idx_tensor, idx)

        ret = {
            "image": imgs,
            "c2w": c2ws,
            "fxfycxcy": intrs,
            "index": torch.cat([idx_tensor, scene_tensor], dim=-1),
            "scene_name": seq,
        }
        if self.load_depth:
            ret["depth"] = depths
            # ret["depth_mask"] = depth_masks
            # we don't pass depth mask for now, since there's a known issue with gsplat 1.5.3 batch rendering
        return ret
