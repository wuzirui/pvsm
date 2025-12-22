import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import random
from rich import print
import traceback

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def read_exr_depth(path: Path):
    depth = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return depth[..., -1]


class GObjaverseDataset(Dataset):
    """
    Gobjaverse dataset loader following the CO3D dataloader format.

    Args:
      - config.gobjaverse.root_dir (str): root directory of gobjaverse data
      - config.gobjaverse.split_file (str): json file listing sequence ids
      - config.training.num_views (int): number of views per sequence
      - config.model.image_tokenizer.image_size (int): target image height (resize keeping aspect)
      - config.gobjaverse.load_depth (bool): whether to load depth (from *_nd.exr)
      - config.gobjaverse.overfitting_mode (bool): whether to overfit on a single sequence
      - config.gobjaverse.allow_resample_on_error (bool): whether to allow resampling on error
    """

    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        self.root_dir = Path(config.gobjaverse.root_dir)
        self.split_file = (
            self.root_dir / "gobjaverse_280k_splits" / config.gobjaverse.split_file
        )
        if split == "val":
            self.split_file = self.root_dir / "gobjaverse_280k_splits" / "val.json"
        self.split = split
        self.num_views = config.training.num_views
        self.image_size = config.model.image_tokenizer.image_size
        self.load_depth = config.gobjaverse.load_depth
        self.overfitting_mode = config.gobjaverse.overfitting_mode
        self.allow_resample_on_error = config.gobjaverse.allow_resample_on_error
        self.min_frame_dist = config.training.view_selector.get('min_frame_dist', 1)
        self.max_frame_dist = config.training.view_selector.get('max_frame_dist', 100)

        # load split list
        with open(self.split_file, "r") as f:
            self.sequence_ids = json.load(f)

        print(f"Loaded {len(self.sequence_ids)} sequences from split={self.split_file}")

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        if self.overfitting_mode:
            idx = 0
        try:
            data = self.load_data(idx)
        except Exception as e:
            if self.allow_resample_on_error:
                print(f"Error loading data for seq={self.sequence_ids[idx]}: {e}...")
                print("See gobjaverse_error.log for details...")
                # append stack trace to local log
                with open("gobjaverse_error.log", "a") as f:
                    f.write(
                        f"Error loading data for seq={self.sequence_ids[idx]}: {e}\n"
                    )
                    f.write(traceback.format_exc())
                return self.__getitem__(random.randint(0, len(self) - 1))
            else:
                raise e
        return data

    def load_data(self, idx):
        """
        Load data for a given index.
        """
        seq = self.sequence_ids[idx]
        seq_path = self.root_dir / seq / "campos_512_v4"
        frame_dirs = sorted([d for d in seq_path.iterdir() if d.is_dir()])

        # 随机选 num_views 个帧
        sel = np.random.choice(len(frame_dirs), self.num_views, replace=False)
        recs = [frame_dirs[i] for i in sel]

        images, intrs, c2ws, depths = [], [], [], []
        for frame_dir in recs:
            frame_id = frame_dir.name
            json_path = frame_dir / f"{frame_id}.json"
            png_path = frame_dir / f"{frame_id}.png"
            nd_path = frame_dir / f"{frame_id}_nd.exr"

            with open(json_path, "r") as f:
                meta = json.load(f)

            # load image
            try:
                img = Image.open(png_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {png_path}: {e}")
                raise e

            orig_w, orig_h = img.size
            new_h = self.image_size
            new_w = int(round(new_h / orig_h * orig_w))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(arr).permute(2, 0, 1).float()

            # intrinsics
            fx = (new_w / 2.0) / np.tan(meta["x_fov"] / 2.0)
            fy = (new_h / 2.0) / np.tan(meta["y_fov"] / 2.0)
            cx, cy = new_w / 2.0, new_h / 2.0
            intr_tensor = torch.tensor([fx, fy, cx, cy], dtype=torch.float)

            # extrinsics
            c2w = np.eye(4)
            c2w[:3, 0] = np.array(meta["x"])
            c2w[:3, 1] = np.array(meta["y"])
            c2w[:3, 2] = np.array(meta["z"])
            c2w[:3, 3] = np.array(meta["origin"])

            c2w = torch.from_numpy(c2w).float()

            images.append(tensor_img)
            intrs.append(intr_tensor)
            c2ws.append(c2w)

            if self.load_depth:
                assert nd_path.exists(), f"nd_path={nd_path} not found"
                depth = read_exr_depth(nd_path).copy()

                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                cond_pos = np.array(meta["origin"])
                cond_cam_dis = np.linalg.norm(cond_pos, 2)
                near = 0.867
                near_distance = max(cond_cam_dis - near, 1e-3)
                mask = depth < near_distance
                mask = dilate_mask(mask, ksize=3, iterations=1)
                depth[mask] = 0

                depth = torch.from_numpy(depth).unsqueeze(0)
                depths.append(depth)

        images = torch.stack(images)
        intrs = torch.stack(intrs)
        c2ws = torch.stack(c2ws)

        idx_tensor = torch.tensor(sel, dtype=torch.long).unsqueeze(-1)
        scene_tensor = torch.full_like(idx_tensor, idx)

        ret = {
            "image": images,
            "c2w": c2ws,
            "fxfycxcy": intrs,
            "scene_name": seq,
            "index": torch.cat([idx_tensor, scene_tensor], dim=-1),
        }
        if self.load_depth and len(depths) > 0:
            ret["depth"] = torch.stack(depths)
        return ret

    def download_and_extract(self, seq):
        """
        Download and extract the data for a given index.
        """
        url = f"https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse_tar/{seq}.tar"
        seq_path = self.root_dir / seq

        # wget
        os.system(f"wget {url} -O {seq_path}.tar")
        os.system(f"tar -xvf {seq_path}.tar -C {(self.root_dir / seq).parent}")
        os.remove(f"{seq_path}.tar")


def dilate_mask(mask: np.ndarray, ksize: int = 3, iterations: int = 1):
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated.astype(bool)
