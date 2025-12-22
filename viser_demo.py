# torchrun --nproc_per_node 1 --standalone viser_demo.py --config-name runs/re10k_lvsm_bs16

import importlib
import threading
from copy import deepcopy
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import torch
import viser
import numpy as np
import nerfview
import time
from tqdm.auto import tqdm
from viser.transforms import SE3, SO3
import hydra
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import random

from utils.metric_utils import apply_float_colormap
from utils.data_utils import create_video_from_frames
from utils.viewer_utils import _rand_se3_np

# # Pre-allocated noise
# noise = torch.rand(1, 8, 3, 256, 256, device="cuda", dtype=torch.float32)


class ViserDemo:
    # ----------------------
    # Utility: Random SE(3)
    # ----------------------
    def _apply_random_se3(self, handle=None):
        self.world_T_np = _rand_se3_np(
            angle_deg=float(self.rot_angle_slider.value), max_t_norm=0.1
        )
        self.world_se3_text.value = "World SE(3):\n" + np.array2string(
            self.world_T_np, precision=3, suppress_small=True
        )
        if self.playmode:
            self.motions.append(self.world_T_np)

        self._dz_reset_state()
        self._refresh_frusta_from_batch()
        if self.viewer is not None:
            self.viewer.rerender(handle)
        print("[World] Applied new random SE(3) to world frame.")

    def _replay(self, handle=None):
        for motion in self.motions:
            self.world_T_np = motion
            self.world_se3_text.value = "World SE(3):\n" + np.array2string(
                self.world_T_np, precision=3, suppress_small=True
            )
            self._dz_reset_state()
            self._refresh_frusta_from_batch()
            if self.viewer is not None:
                self.viewer.rerender(handle)
            time.sleep(0.1)

    # -------------
    # Data handling
    # -------------
    def resample_batch(self, handle=None):
        if self.random_sample:
            index_id = random.randint(0, len(self.dataset) - 1)
        else:
            index_id = self.id_input.value
        batch = self.dataset[index_id]
        print(
            f"Re-sampled data to index {index_id}, with {batch['image'].shape[0]} views"
        )

        with self.batch_lock:
            self.batch = edict(
                {
                    k: v[None].to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            )
            # self.batch["noise"] = noise
            self.current_id_text.value = f"{index_id}"
            # DZ state reset (anchor无效了)
            self.dz_enabled = False
            self.dz_initialized = False
            self.dz_anchor = edict(Z0=None, C0=None, R0=None, fx0=None, fy0=None)

        try:
            c2w0 = self.batch["c2w"][0, 0].detach().cpu().numpy()
            self._update_viewer_camera_to(c2w0)
        except Exception as e:
            print(f"[viewer camera update skipped] {e}")

        self._refresh_frusta_from_batch()

        # reset dolly zoom state
        self._dz_reset_state()

        if self.viewer is not None:
            self.viewer.rerender(handle)

    def get_viewer(self):
        return self.viewer

    def update_random(self, handle=None):
        if self.random_sample_checkbox.value:
            self.random_sample = True
            self.id_input.disabled = True
        else:
            self.random_sample = False
            self.id_input.disabled = False
            self.id_input.value = self.current_id_text.value
        self.resample_batch(handle)

    def update_num_input_views(self, handle=None):
        num_input_views = self.num_input_views.value
        num_views = num_input_views + 1
        self.dataset.num_views = num_views
        self.model.process_data.config.training.num_views = num_views
        self.model.process_data.config.training.num_input_views = num_input_views

        current_random_sample = self.random_sample
        self.random_sample = False
        self.id_input.value = self.current_id_text.value
        self.random_sample = current_random_sample

        self._refresh_frusta_from_batch()
        self._dz_reset_state()

    # ----------------------
    # Helper: select output channel -> numpy image [0,1]
    # ----------------------
    def _result_to_numpy(self, result, use_center_dot: bool = False) -> np.ndarray:
        """
        Convert the model's result according to the current GUI channel setting into a [H, W, 3] numpy array (float32, values in [0,1]).
        The logic is identical to that in render_fn, used for both interactive rendering and various exports.
        """
        img_tensor = result["render"]
        _, _, _, h, w = img_tensor.shape

        # 1) point cloud image
        if self.show_point_cloud_image_checkbox.value:
            pc_img = result["target"]["pc_image"]  # [B,1,C,H,W]
            img = pc_img[0, 0].permute(1, 2, 0).float().cpu().numpy()

        # 2) depth colormap
        elif self.show_depth_image_checkbox.value:
            depth = result["target"]["pc_depth"][0, 0, None].permute(0, 3, 1, 2)
            depth = depth / (depth.amax() + 1e-6)
            img = apply_float_colormap(depth)[0].float().cpu().permute(1, 2, 0).numpy()

        # 3) Plücker moment
        elif self.show_plucker_moment_checkbox.value:
            ray_o = result["target"]["ray_o"][0, 0].detach().cpu()
            ray_d = result["target"]["ray_d"][0, 0].detach().cpu()
            plucker_moment = torch.cross(ray_o, ray_d, dim=0).clamp(-1, 1) * 0.5 + 0.5
            img = plucker_moment.permute(1, 2, 0).float().cpu().numpy()

        # 4) default render
        else:
            img = img_tensor[0, 0].permute(1, 2, 0).float().cpu().numpy()

        # center red dot (only used for interactive rendering, not used for exports)
        if use_center_dot and self.render_center_red_dot_checkbox.value:
            img[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = [1, 0, 0]

        return img

    # ----------------------
    # Dolly Zoom State Utils
    # ----------------------
    def _dz_reset_state(self):
        self.dz_enabled = False
        self.dz_initialized = False
        self.dz_anchor = edict(Z0=None, C0=None, R0=None, fx0=None, fy0=None)
        self.dz_restore = edict(C=None, R=None, fov_deg=None, roll_deg=None)
        self.dz_restore_pose_next = False
        self.dz_state_text.value = "OFF"

    def _toggle_dz(self, handle=None):
        # toggle: OFF -> ON / ON -> OFF
        if not self.dz_enabled:
            # enable: capture the anchor at the current view, cancel any previous "next frame restore" action
            self.dz_enabled = True
            self.dz_initialized = False
            self.dz_restore_pose_next = False
            self.dz_state_text.value = "ON (capturing anchor...)"
            if self.viewer is not None:
                self.viewer.rerender(handle)
        else:
            # disable: restore to the view when the anchor was captured
            self.dz_enabled = False
            self.dz_initialized = False
            if self.dz_restore.fov_deg is not None:
                self.fov_slider.value = float(self.dz_restore.fov_deg)
            if self.dz_restore.roll_deg is not None:
                self.roll_slider.value = float(self.dz_restore.roll_deg)
            self.dz_restore_pose_next = True
            self.dz_state_text.value = "OFF (restored)"
            if self.viewer is not None:
                self.viewer.rerender(handle)

    # ----------------------
    # Blocking Dolly Export
    # ----------------------
    def _export_dolly_sequence(self, handle=None):
        """
        Blocking export dolly zoom sequence:
        FOV = 10, 20, ..., 120, render one frame for each FOV, save to experiments/render/dolly-zoom-demo.
        Use current world_T / global_scene_scale / roll.
        If there is no anchor, first run a frame to capture the anchor with the current view.
        """
        out_dir = Path("experiments/render/dolly-zoom-demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        # fovs = list(range(int(self.min_fov_number.value), int(self.max_fov_number.value) + 1, 1))
        fovs = np.linspace(
            self.min_fov_number.value,
            self.max_fov_number.value,
            self.total_frames_number.value,
        ).tolist()
        print(fovs)

        print(f"[DZ export] Start, saving to: {out_dir}")

        # --- 1) take a batch snapshot (dataset units) ---
        with self.batch_lock:
            base = self.batch
            batch_base = edict({k: v for k, v in base.items()})
            if isinstance(base.get("c2w"), torch.Tensor):
                batch_base["c2w"] = base["c2w"].clone()
            if isinstance(base.get("fxfycxcy"), torch.Tensor):
                batch_base["fxfycxcy"] = base["fxfycxcy"].clone()

        dev = batch_base["c2w"].device
        dtype = batch_base["c2w"].dtype
        T = torch.tensor(self.world_T_np, device=dev, dtype=dtype)

        # apply world_T to all contexts
        batch_base["c2w"][0] = torch.einsum("ij,njk->nik", T, batch_base["c2w"][0])

        # resolution & numerical scale
        h = batch_base["image"].shape[-2]
        w = batch_base["image"].shape[-1]
        numerical_scale = 10 ** float(self.numerical_scale_slider.value)

        # --- 2) determine the baseline camera pose (current viewer view) ---
        if getattr(self, "last_camera_state", None) is not None:
            c2w0 = torch.tensor(self.last_camera_state.c2w, device=dev, dtype=dtype)
            c2w0[:3, 3] = c2w0[:3, 3] / self.global_scene_scale.value
        else:
            c2w0 = batch_base["c2w"][0, -1].clone()

        # apply world_T
        c2w0 = T @ c2w0

        # roll
        theta0 = float(self.roll_slider.value) * math.pi / 180.0
        ct0, st0 = math.cos(theta0), math.sin(theta0)
        R_roll0 = torch.eye(4, dtype=dtype, device=dev)
        R_roll0[0, 0] = ct0
        R_roll0[0, 1] = -st0
        R_roll0[1, 0] = st0
        R_roll0[1, 1] = ct0
        c2w0 = c2w0 @ R_roll0

        # --- 3) anchor: if already exists, use it; otherwise, first run a frame to capture the anchor with the current view ---
        if self.dz_initialized and (self.dz_anchor.Z0 is not None):
            Z0 = float(self.dz_anchor.Z0)
            C0 = self.dz_anchor.C0.to(dev)
            R0 = self.dz_anchor.R0.to(dev)
            fx0 = float(self.dz_anchor.fx0)
        else:
            fov0 = float(self.fov_slider.value)
            fov0_rad = fov0 * math.pi / 180.0
            aspect0 = float(self.aspect_ratio_slider.value)

            fy0 = h / (2.0 * math.tan(0.5 * fov0_rad))
            fx0 = aspect0 * fy0

            batch_anchor = edict({k: v for k, v in batch_base.items()})
            batch_anchor["fxfycxcy"] = batch_anchor["fxfycxcy"].clone()
            batch_anchor["c2w"] = batch_anchor["c2w"].clone()
            batch_anchor["fxfycxcy"][0, -1, 0] = fx0
            batch_anchor["fxfycxcy"][0, -1, 1] = fy0
            batch_anchor["c2w"][:, -1] = c2w0
            batch_anchor["c2w"] *= numerical_scale

            with torch.no_grad(), torch.autocast(
                enabled=self.config.training.use_amp,
                device_type="cuda",
                dtype=self.amp_dtype_mapping[self.config.training.amp_dtype],
            ):
                result0 = self.model(
                    batch_anchor, has_target_image=False, compute_pc_image=True
                )
            z_center_model = (
                result0["target"]["pc_depth"][0, 0, h // 2, w // 2, 0]
                .detach()
                .float()
                .item()
            )
            Z0 = z_center_model / (numerical_scale + 1e-12)
            C0 = c2w0[:3, 3].detach().clone()
            R0 = c2w0[:3, :3].detach().clone()

            # update dz_anchor, so that it can be reused for interactive rendering
            self.dz_anchor.Z0 = Z0
            self.dz_anchor.C0 = C0.detach().cpu()
            self.dz_anchor.R0 = R0.detach().cpu()
            self.dz_anchor.fx0 = float(fx0)
            self.dz_anchor.fy0 = float(fy0)
            self.dz_initialized = True

            print(
                f"[DZ export] Captured anchor: Z0={Z0:.4f}, fx0={fx0:.2f} (from current view)"
            )

        # --- 4) loop over FOVs, do dolly zoom, and save the rendered frames ---
        bar = tqdm(total=len(fovs), desc="Dolly zoom export", leave=True)
        frames = []
        for fov in fovs:
            fov_deg = float(fov)
            fov_rad = fov_deg * math.pi / 180.0
            aspect = float(self.aspect_ratio_slider.value)

            fy = h / (2.0 * math.tan(0.5 * fov_rad))
            fx = aspect * fy

            s = fx / (fx0 + 1e-12)
            delta = Z0 * (1.0 - s)
            forward = R0[:, 2]
            if not torch.is_tensor(forward):
                forward = torch.tensor(forward, device=dev, dtype=dtype)
            C = C0 + delta * forward

            c2w = torch.eye(4, dtype=dtype, device=dev)
            c2w[:3, :3] = R0
            c2w[:3, 3] = C

            batch_frame = edict({k: v for k, v in batch_base.items()})
            batch_frame["fxfycxcy"] = batch_frame["fxfycxcy"].clone()
            batch_frame["c2w"] = batch_frame["c2w"].clone()

            batch_frame["fxfycxcy"][0, -1, 0] = fx
            batch_frame["fxfycxcy"][0, -1, 1] = fy
            batch_frame["c2w"][:, -1] = c2w
            batch_frame["c2w"] *= numerical_scale

            with torch.no_grad(), torch.autocast(
                enabled=self.config.training.use_amp,
                device_type="cuda",
                dtype=self.amp_dtype_mapping[self.config.training.amp_dtype],
            ):
                result = self.model(
                    batch_frame, has_target_image=False, compute_pc_image=True
                )
            img = self._result_to_numpy(result, use_center_dot=False)
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            frames.append(img_uint8)

            bar.update(1)

        create_video_from_frames(frames, str(out_dir / "rendered_video.mp4"))
        bar.close()
        print(f"[DZ export] Done, saved {len(fovs)} frames to {out_dir}")
        print(f"[DZ export] Saved video to {out_dir / 'rendered_video.mp4'}")

    # Helper: current dataset camera (with DZ / roll / gauge)
    # ----------------------
    def _get_current_dataset_c2w(self):
        """
        Reconstruct the camera geometry transformation of render_fn based on last_camera_state + current slider + DZ state,
        and get the c2w in the dataset coordinate system.
        """
        if self.last_camera_state is None:
            print("[CameraPath] No camera_state yet; move the viewer once first.")
            return None

        with self.batch_lock:
            if "c2w" not in self.batch or "image" not in self.batch:
                print("[CameraPath] Batch is not ready.")
                return None

            dev = self.batch["c2w"].device
            dtype = self.batch["c2w"].dtype
            h = self.batch["image"].shape[-2]
            w = self.batch["image"].shape[-1]

        # 1) start from the outer camera of the viewer
        c2w_view = torch.tensor(self.last_camera_state.c2w, device=dev, dtype=dtype)

        # remove global_scene_scale
        scale = float(self.global_scene_scale.value)
        if scale == 0.0:
            scale = 1.0
        c2w_view[:3, 3] = c2w_view[:3, 3] / scale

        # apply world_T
        T_np = self.world_T_np.astype(np.float32)
        T = torch.tensor(T_np, device=dev, dtype=dtype)
        c2w = T @ c2w_view  # world gauge camera

        # 2) FOV -> fx/fy
        fov_deg = float(self.fov_slider.value)
        fov_rad = fov_deg * math.pi / 180.0
        aspect = float(self.aspect_ratio_slider.value)

        new_fy = h / (2.0 * math.tan(0.5 * fov_rad))
        new_fx = aspect * new_fy

        # 3) Roll
        roll_deg = float(self.roll_slider.value)
        theta = roll_deg * math.pi / 180.0
        ct, st = math.cos(theta), math.sin(theta)
        R_roll = torch.eye(4, dtype=dtype, device=dev)
        R_roll[0, 0] = ct
        R_roll[0, 1] = -st
        R_roll[1, 0] = st
        R_roll[1, 1] = ct
        c2w = c2w @ R_roll

        # 4) if currently in the "first frame restore state after DZ is disabled", keep the same as render_fn
        if (
            getattr(self, "dz_restore_pose_next", False)
            and (self.dz_restore.C is not None)
            and (self.dz_restore.R is not None)
        ):
            C_restore = (
                self.dz_restore.C.to(device=dev, dtype=dtype)
                if isinstance(self.dz_restore.C, torch.Tensor)
                else torch.tensor(self.dz_restore.C, device=dev, dtype=dtype)
            )
            R_restore = (
                self.dz_restore.R.to(device=dev, dtype=dtype)
                if isinstance(self.dz_restore.R, torch.Tensor)
                else torch.tensor(self.dz_restore.R, device=dev, dtype=dtype)
            )
            c2w[:3, :3] = R_restore
            c2w[:3, 3] = C_restore
            # note: here we don't change dz_restore_pose_next, this function only reads the state

        # 5) if Dolly Zoom is enabled and initialized, reuse the same Z0 / forward logic
        if (
            getattr(self, "dz_enabled", False)
            and getattr(self, "dz_initialized", False)
            and (self.dz_anchor.Z0 is not None)
        ):
            Z0 = float(self.dz_anchor.Z0)
            fx0 = float(self.dz_anchor.fx0)
            s = float(new_fx) / (fx0 + 1e-12)
            delta = Z0 * (1.0 - s)

            # anchor is R0 / C0 in the world gauge
            C0 = (
                self.dz_anchor.C0.to(device=dev, dtype=dtype)
                if isinstance(self.dz_anchor.C0, torch.Tensor)
                else torch.tensor(self.dz_anchor.C0, device=dev, dtype=dtype)
            )
            R0 = (
                self.dz_anchor.R0.to(device=dev, dtype=dtype)
                if isinstance(self.dz_anchor.R0, torch.Tensor)
                else torch.tensor(self.dz_anchor.R0, device=dev, dtype=dtype)
            )
            forward = R0[:, 2]  # (3,)

            C_new = C0 + delta * forward
            c2w[:3, 3] = C_new

        # 6) transform the c2w in the world gauge back to the dataset coordinate system
        T_inv_np = np.linalg.inv(T_np)
        c2w_world_np = c2w.detach().cpu().numpy().astype(np.float32)
        c2w_dataset_np = T_inv_np @ c2w_world_np

        return c2w_dataset_np

    # ----------------------
    # Camera Path Keyframes
    # ----------------------
    def _update_keyframe_summary(self):
        """Update the text listing of all keyframes."""
        if not getattr(self, "keyframes", []):
            self.kf_summary_text.value = "(none)"
            return

        lines = []
        for i, kf in enumerate(self.keyframes):
            c2w = kf["c2w"]
            C = c2w[:3, 3]
            fov = kf["fov_deg"]
            roll = kf["roll_deg"]
            num_log10 = kf.get("numerical_scale_log10", None)
            aspect = kf.get("aspect_ratio", None)

            parts = [
                f"{i:02d}: fov={fov:.1f}°",
                f"roll={roll:.1f}°",
                f"C=({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})",
            ]
            if num_log10 is not None:
                parts.append(f"log10(s)={num_log10:.2f}")
            if aspect is not None:
                parts.append(f"ar={aspect:.2f}")

            lines.append(", ".join(parts))

        self.kf_summary_text.value = "\n".join(lines)

    def _add_keyframe(self, handle=None):
        """
        Transform the current camera (considering DZ / roll / world_T / global_scene_scale)
        and transform it back to the dataset coordinate system, then save it as a keyframe, along with the numerical scale (log10).
        """
        c2w_dataset = self._get_current_dataset_c2w()
        if c2w_dataset is None:
            return

        fov_deg = float(self.fov_slider.value)
        roll_deg = float(self.roll_slider.value)
        aspect = float(self.aspect_ratio_slider.value)
        num_log10 = float(self.numerical_scale_slider.value)  # between 0 and 4

        if not hasattr(self, "keyframes"):
            self.keyframes = []

        self.keyframes.append(
            {
                "c2w": c2w_dataset,  # 4x4, dataset coordinate system
                "fov_deg": fov_deg,
                "roll_deg": roll_deg,
                "numerical_scale_log10": num_log10,
                "aspect_ratio": aspect,
            }
        )
        print(
            f"[CameraPath] Added keyframe #{len(self.keyframes)-1}: "
            f"fov={fov_deg:.1f}, roll={roll_deg:.1f}, log10(s)={num_log10:.2f}, aspect={aspect:.2f}"
        )
        self._update_keyframe_summary()

    def _clear_keyframes(self, handle=None):
        """Clear all keyframes."""
        self.keyframes = []
        self._update_keyframe_summary()
        print("[CameraPath] Cleared all keyframes.")

    def _slerp_so3(self, R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
        """
        Do geodesic interpolation on SO(3): R(t) = R0 * exp( t * log(R0^{-1} * R1) ).
        Use the log/exp functions in viser.transforms.SO3.
        """
        # single 3x3 matrix
        so3_0 = SO3.from_matrix(R0.astype(np.float64))
        so3_1 = SO3.from_matrix(R1.astype(np.float64))

        delta = so3_0.inverse().multiply(so3_1)
        tangent = delta.log()  # (3,)

        so3_t = so3_0.multiply(SO3.exp(t * tangent))
        return so3_t.as_matrix().astype(np.float32)

    def _export_camera_path(self, handle=None):
        """
        Blocking export camera path sequence:
        - In the viewer, drag the camera / adjust fov / roll, add a series of keyframes;
        - Linearly interpolate C / fov / roll, do geodesic interpolation on SO(3) for R;
        - Render using current world_T / global_scene_scale / numerical scale;
        - Save to experiments/render/camera-path-demo.
        """
        if not getattr(self, "keyframes", None) or len(self.keyframes) < 2:
            print("[CameraPath] Need at least 2 keyframes to export.")
            return

        out_dir = Path("experiments/render/camera-path-demo")
        out_dir.mkdir(parents=True, exist_ok=True)

        frames_per_segment = int(self.kf_frames_per_segment.value)
        if frames_per_segment <= 0:
            print("[CameraPath] Frames per segment must be >= 1.")
            return

        print(
            f"[CameraPath] Exporting path with {len(self.keyframes)} keyframes, "
            f"{frames_per_segment} frames / segment."
        )

        # 1) take a batch snapshot
        with self.batch_lock:
            base = self.batch
            batch_base = edict({k: v for k, v in base.items()})
            if isinstance(base.get("c2w"), torch.Tensor):
                batch_base["c2w"] = base["c2w"].clone()
            if isinstance(base.get("fxfycxcy"), torch.Tensor):
                batch_base["fxfycxcy"] = base["fxfycxcy"].clone()

        dev = batch_base["c2w"].device
        dtype = batch_base["c2w"].dtype

        T = torch.tensor(self.world_T_np, device=dev, dtype=dtype)

        # apply world_T to all contexts
        batch_base["c2w"][0] = torch.einsum("ij,njk->nik", T, batch_base["c2w"][0])

        h = batch_base["image"].shape[-2]
        w = batch_base["image"].shape[-1]

        # 2) prepare keyframe data (all in the dataset coordinate system)
        Ks = self.keyframes
        num_kf = len(Ks)
        positions = np.stack([kf["c2w"][:3, 3] for kf in Ks], axis=0)  # (N, 3)
        rotations = np.stack([kf["c2w"][:3, :3] for kf in Ks], axis=0)  # (N, 3, 3)
        fovs = np.array([kf["fov_deg"] for kf in Ks], dtype=np.float32)  # (N,)
        rolls = np.array([kf["roll_deg"] for kf in Ks], dtype=np.float32)  # (N,)

        # numerical scale: log10(s)
        # for old keyframes that don't have this field, take a default value from the current slider
        default_log10 = float(self.numerical_scale_slider.value)
        num_logs = np.array(
            [float(kf.get("numerical_scale_log10", default_log10)) for kf in Ks],
            dtype=np.float32,
        )  # (N,)

        default_aspect = float(self.aspect_ratio_slider.value)
        aspects = np.array(
            [float(kf.get("aspect_ratio", default_aspect)) for kf in Ks],
            dtype=np.float32,
        )

        # 3) loop over each segment of keyframes, do interpolation and render
        total_segments = num_kf - 1
        total_frames = total_segments * frames_per_segment - (total_segments - 1)
        bar = tqdm(total=total_frames, desc="Camera path export", leave=True)
        frames = []

        use_kf_scale = getattr(self, "kf_enable_numerical_scale_checkbox", None)
        use_kf_scale = bool(use_kf_scale.value) if use_kf_scale is not None else False
        for seg_idx in range(total_segments):
            C0 = positions[seg_idx]
            C1 = positions[seg_idx + 1]
            R0 = rotations[seg_idx]
            R1 = rotations[seg_idx + 1]
            fov0 = float(fovs[seg_idx])
            fov1 = float(fovs[seg_idx + 1])
            roll0 = float(rolls[seg_idx])
            roll1 = float(rolls[seg_idx + 1])
            num_log0 = float(num_logs[seg_idx])
            num_log1 = float(num_logs[seg_idx + 1])
            aspect0 = float(aspects[seg_idx])
            aspect1 = float(aspects[seg_idx + 1])

            for step in range(frames_per_segment):
                # avoid duplicate sampling keyframes: skip the 0th frame of each segment
                if seg_idx > 0 and step == 0:
                    continue

                if frames_per_segment == 1:
                    t = 0.0
                else:
                    t = step / (frames_per_segment - 1)

                # position & scalar linear interpolation
                C_t = (1.0 - t) * C0 + t * C1
                fov_t = (1.0 - t) * fov0 + t * fov1
                roll_t = (1.0 - t) * roll0 + t * roll1

                if use_kf_scale:
                    num_log_t = (1.0 - t) * num_log0 + t * num_log1
                else:
                    num_log_t = float(self.numerical_scale_slider.value)

                numerical_scale = 10**num_log_t
                aspect_t = (1.0 - t) * aspect0 + t * aspect1
                # rotate using geodesic interpolation
                R_t = self._slerp_so3(R0, R1, t)

                # assemble the c2w of the current frame (dataset coordinate system)
                c2w_np = np.eye(4, dtype=np.float32)
                c2w_np[:3, :3] = R_t
                c2w_np[:3, 3] = C_t

                c2w = torch.tensor(c2w_np, device=dev, dtype=dtype)
                # apply world_T
                c2w = T @ c2w

                # roll
                theta = roll_t * math.pi / 180.0
                ct, st = math.cos(theta), math.sin(theta)
                R_roll = torch.eye(4, dtype=dtype, device=dev)
                R_roll[0, 0] = ct
                R_roll[0, 1] = -st
                R_roll[1, 0] = st
                R_roll[1, 1] = ct
                c2w = c2w @ R_roll

                # fov -> fx, fy
                # fov + aspect -> fx, fy
                fov_rad = fov_t * math.pi / 180.0
                fy = h / (2.0 * math.tan(0.5 * fov_rad))
                fx = aspect_t * fy

                # assemble the batch
                batch_frame = edict({k: v for k, v in batch_base.items()})
                batch_frame["fxfycxcy"] = batch_frame["fxfycxcy"].clone()
                batch_frame["c2w"] = batch_frame["c2w"].clone()

                batch_frame["fxfycxcy"][0, -1, 0] = fx
                batch_frame["fxfycxcy"][0, -1, 1] = fy
                batch_frame["c2w"][:, -1] = c2w

                batch_frame["c2w"] *= numerical_scale

                # render
                with torch.no_grad(), torch.autocast(
                    enabled=self.config.training.use_amp,
                    device_type="cuda",
                    dtype=self.amp_dtype_mapping[self.config.training.amp_dtype],
                ):
                    result = self.model(
                        batch_frame, has_target_image=False, compute_pc_image=True
                    )

                img = self._result_to_numpy(result, use_center_dot=False)
                img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                frames.append(img_uint8)

                bar.update(1)

        create_video_from_frames(frames, str(out_dir / "camera_path.mp4"))
        bar.close()
        print(
            f"[CameraPath] Done, saved {len(frames)} frames to {out_dir}, "
            f"video: {out_dir / 'camera_path.mp4'}"
        )

    # ----------------------
    # Save / Load Session (batch + keyframes)
    # ----------------------
    def _get_session_path(self) -> Path:
        """Take the name from the text box, if empty, use the default value 'default'."""
        name = (
            self.session_name_text.value.strip()
            if hasattr(self, "session_name_text")
            else ""
        )
        if not name:
            name = "default"
        session_dir = Path("experiments/session")
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"{name}.pt"

    def _save_session(self, handle=None):
        """
        Save the current batch (moved to CPU) and keyframes to a pt file.
        """
        path = self._get_session_path()
        with self.batch_lock:
            batch_cpu = {}
            for k, v in self.batch.items():
                if isinstance(v, torch.Tensor):
                    batch_cpu[k] = v.detach().cpu()
                else:
                    batch_cpu[k] = v
            payload = {
                "batch": batch_cpu,
                "keyframes": getattr(self, "keyframes", []),
            }

        torch.save(payload, path)
        print(f"[Session] Saved batch + keyframes to: {path}")

    def _load_session(self, handle=None):
        """
        Restore the batch (moved back to self.device) and keyframes from a pt file.
        """
        path = self._get_session_path()
        if not path.exists():
            print(f"[Session] File not found: {path}")
            return

        payload = torch.load(path, map_location=self.device)
        batch_loaded = payload.get("batch", {})
        keyframes_loaded = payload.get("keyframes", [])

        with self.batch_lock:
            batch_dev = {}
            for k, v in batch_loaded.items():
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.to(self.device)
                else:
                    batch_dev[k] = v
            self.batch = edict(batch_dev)

            # restore keyframes
            self.keyframes = keyframes_loaded

        # refresh keyframe list and frusta / view
        if hasattr(self, "_update_keyframe_summary"):
            self._update_keyframe_summary()

        self._refresh_frusta_from_batch()
        try:
            c2w0 = self.batch["c2w"][0, 0].detach().cpu().numpy()
            self._update_viewer_camera_to(c2w0)
        except Exception as e:
            print(f"[Session] Camera reset failed after load: {e}")

        if self.viewer is not None:
            self.viewer.rerender(handle)

        print(f"[Session] Loaded batch + keyframes from: {path}")

    # ----------------------
    # Init
    # ----------------------
    def __init__(self, config):
        self.viewer = None
        self.random_sample = True
        self.id_input = None
        self.config = config
        self.batch_lock = threading.RLock()
        self.last_camera_state = None

        # Camera path keyframes
        self.keyframes = []

        os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

        torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
        torch.backends.cudnn.allow_tf32 = config.training.use_tf32

        self.amp_dtype_mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "tf32": torch.float32,
        }

        dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
        module, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module).__dict__[class_name]
        self.dataset = Dataset(config, split="test")

        module, class_name = config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module).__dict__[class_name]
        self.device = "cuda"
        self.model = LVSM(config).to(self.device)
        self.model.load_ckpt(config.training.checkpoint_dir)

        server = viser.ViserServer(verbose=False)
        self.server = server
        self._clients = set()
        self.frusta = {}
        self.world_T_np = np.eye(4, dtype=np.float32)

        self.show_point_cloud_image_checkbox = server.gui.add_checkbox(
            "Show Point Cloud Image", initial_value=False
        )
        self.show_depth_image_checkbox = server.gui.add_checkbox(
            "Show Depth Image", initial_value=False
        )
        self.show_plucker_moment_checkbox = server.gui.add_checkbox(
            "Show Plucker Moment",
            initial_value=False,
            visible=("LVSM" in self.config.model.class_name),
        )

        self.random_sample_checkbox = server.gui.add_checkbox(
            "Random Sample", initial_value=True
        )
        self.random_sample_checkbox.on_update(self.update_random)

        self.id_input = server.gui.add_number(
            "ID", initial_value=0, min=0, max=len(self.dataset) - 1, disabled=True
        )
        self.id_input.on_update(self.resample_batch)
        self.current_id_text = server.gui.add_text(
            "Current ID", initial_value="0", disabled=True
        )

        self.global_scene_scale = server.gui.add_slider(
            "Global Scene Scale", min=1e-2, max=1e2, initial_value=10, step=1e-2
        )
        self.num_input_views = server.gui.add_slider(
            "Num Input Views", min=1, max=30, initial_value=3, step=1
        )
        self.fov_slider = server.gui.add_slider(
            "FOV", min=10.0, max=120.0, initial_value=60.0, step=0.1
        )
        self.aspect_ratio_slider = server.gui.add_slider(
            "Aspect Ratio (fx/fy)", min=0.25, max=4.0, initial_value=1.0, step=0.01
        )
        self.roll_slider = server.gui.add_slider(
            "Roll", min=-360.0, max=360.0, initial_value=0.0, step=0.1
        )
        self.numerical_scale_slider = server.gui.add_slider(
            "Numerical Scale (Log-level)", min=0, max=4, initial_value=0, step=0.1
        )
        self.num_input_views.on_update(self.update_num_input_views)
        button = server.gui.add_button("Re-sample Data")
        button.on_click(self.resample_batch)

        # Dolly Zoom toggle & state
        with server.gui.add_folder("Dolly Zoom Kit", expand_by_default=True):
            self.render_center_red_dot_checkbox = server.gui.add_checkbox(
                "Show Center", initial_value=False
            )
            self.dz_toggle_btn = server.gui.add_button("Toggle Dolly Zoom")
            self.dz_toggle_btn.on_click(self._toggle_dz)
            self.dz_state_text = server.gui.add_text(
                "Dolly Zoom Status", initial_value="OFF", disabled=True
            )

        # export dolly sequence button (blocking)
        with server.gui.add_folder("Export Dolly Sequence", expand_by_default=False):
            self.min_fov_number = server.gui.add_number(
                "Min FOV", initial_value=10.0, min=1.0, max=120.0, step=0.1
            )
            self.max_fov_number = server.gui.add_number(
                "Max FOV", initial_value=120.0, min=1.0, max=120.0, step=0.1
            )
            self.total_frames_number = server.gui.add_number(
                "Total Frames", initial_value=60, min=1, max=1000, step=1
            )
            self.export_dz_btn = server.gui.add_button("Export Dolly Sequence")
            self.export_dz_btn.on_click(self._export_dolly_sequence)

            # Random SE(3) button
        with server.gui.add_folder("Random Gauge", expand_by_default=False):
            self.rot_angle_slider = server.gui.add_slider(
                "World Rot Angle (deg)",
                min=0.0,
                max=180.0,
                initial_value=90.0,
                step=1.0,
            )
            self.random_se3_btn = server.gui.add_button("Random SE(3) (apply to world)")
            self.random_se3_btn.on_click(self._apply_random_se3)
            self.rot_angle_slider.on_update(self._apply_random_se3)
            self.world_se3_text = server.gui.add_text(
                "World SE(3)", initial_value="Identity", disabled=True
            )
            self.random_se3_play_btn = server.gui.add_button("Play Random SE(3)")
            self.motions = []
            self.playmode = False

            def _play_random_se3(_):
                self.motions = []
                self.playmode = True
                for i in range(180):
                    self.rot_angle_slider.value = i
                    time.sleep(0.1)
                self.playmode = False

            self.random_se3_play_btn.on_click(_play_random_se3)
            self.replay_btn = server.gui.add_button("Replay")
            self.replay_btn.on_click(self._replay)

        # Camera path keyframes
        with server.gui.add_folder("Camera Path", expand_by_default=False):
            self.kf_add_button = server.gui.add_button("Add Keyframe (current view)")
            self.kf_add_button.on_click(self._add_keyframe)

            self.kf_clear_button = server.gui.add_button("Clear Keyframes")
            self.kf_clear_button.on_click(self._clear_keyframes)

            self.kf_frames_per_segment = server.gui.add_number(
                "Frames per segment", initial_value=30, min=1, max=1000, step=1
            )

            # whether to allow numerical scale to change with keyframe
            self.kf_enable_numerical_scale_checkbox = server.gui.add_checkbox(
                "Enable Numerical Scale Animation", initial_value=False
            )

            self.kf_summary_text = server.gui.add_text(
                "Keyframe List",
                initial_value="(none)",
                multiline=True,
                disabled=True,
            )

            self.kf_export_button = server.gui.add_button("Export Camera Path")
            self.kf_export_button.on_click(self._export_camera_path)

        # Save / Load Session
        with server.gui.add_folder("Save / Load Session", expand_by_default=False):
            # text box: if empty, use "default"
            self.session_name_text = server.gui.add_text(
                "Session Name (empty = default)", initial_value="", disabled=False
            )
            self.save_session_button = server.gui.add_button(
                "Save Session (batch + keyframes)"
            )
            self.save_session_button.on_click(self._save_session)

            self.load_session_button = server.gui.add_button("Load Session")
            self.load_session_button.on_click(self._load_session)

        if self.config.training.dataset_name == "data.wai.WaiDataset":
            with server.gui.add_folder("WAI Query"):
                name_text = server.gui.add_text(
                    "Scene Name", initial_value="", disabled=False
                )
                id_lb = server.gui.add_text("Scene ID", initial_value="", disabled=True)

                def query_name(_):
                    name = name_text.value
                    try:
                        id = self.dataset.sequence_list.index(name)
                    except ValueError:
                        id = -1
                    id_lb.value = f"{id}"

                name_text.on_update(query_name)

        self._dz_reset_state()
        self.resample_batch()

        # --------------------------
        # Render function (interactive)
        # --------------------------
        def render_fn(
            camera_state: nerfview.CameraState,
            render_tab_state: nerfview.RenderTabState,
        ) -> np.ndarray:
            self.last_camera_state = camera_state

            with self.batch_lock:
                base = self.batch
                batch = edict({k: v for k, v in base.items()})
                if isinstance(base.get("c2w"), torch.Tensor):
                    batch["c2w"] = base["c2w"].clone()
                if isinstance(base.get("fxfycxcy"), torch.Tensor):
                    batch["fxfycxcy"] = base["fxfycxcy"].clone()

            T = torch.tensor(
                self.world_T_np, device=batch["c2w"].device, dtype=batch["c2w"].dtype
            )
            batch["c2w"][0] = torch.einsum("ij,njk->nik", T, batch["c2w"][0])

            c2w = torch.tensor(camera_state.c2w).to(batch["c2w"])
            c2w[:3, 3] = c2w[:3, 3] / self.global_scene_scale.value
            c2w = T @ c2w

            w = batch["image"].shape[-1]
            h = batch["image"].shape[-2]

            fov_deg = float(self.fov_slider.value)
            fov_rad = fov_deg * math.pi / 180.0
            aspect = float(self.aspect_ratio_slider.value)

            new_fy = h / (2.0 * math.tan(0.5 * fov_rad))  # vertical FOV
            new_fx = aspect * new_fy  # fx/fy = aspect

            theta = float(self.roll_slider.value) * math.pi / 180.0
            ct, st = math.cos(theta), math.sin(theta)
            R_roll = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
            R_roll[0, 0] = ct
            R_roll[0, 1] = -st
            R_roll[1, 0] = st
            R_roll[1, 1] = ct
            c2w = c2w @ R_roll

            if (
                self.dz_restore_pose_next
                and (self.dz_restore.C is not None)
                and (self.dz_restore.R is not None)
            ):
                c2w[:3, :3] = self.dz_restore.R
                c2w[:3, 3] = self.dz_restore.C
                self.dz_restore_pose_next = False

            if (
                self.dz_enabled
                and self.dz_initialized
                and (self.dz_anchor.Z0 is not None)
            ):
                s = float(new_fx) / float(self.dz_anchor.fx0 + 1e-12)
                delta = float(self.dz_anchor.Z0) * (1.0 - s)
                base_forward = self.dz_anchor.R0[:, 2]
                if not torch.is_tensor(base_forward):
                    base_forward = torch.tensor(
                        base_forward, dtype=c2w.dtype, device=c2w.device
                    )
                C_new = self.dz_anchor.C0 + delta * base_forward
                c2w[:3, 3] = C_new

            batch["fxfycxcy"][0, -1, 0] = new_fx
            batch["fxfycxcy"][0, -1, 1] = new_fy
            batch["c2w"][:, -1] = c2w

            numerical_scale = 10 ** float(self.numerical_scale_slider.value)
            batch["c2w"] *= numerical_scale

            with torch.no_grad(), torch.autocast(
                enabled=self.config.training.use_amp,
                device_type="cuda",
                dtype=self.amp_dtype_mapping[self.config.training.amp_dtype],
            ):
                result = self.model(
                    batch, has_target_image=False, compute_pc_image=True
                )

                if self.dz_enabled and not self.dz_initialized:
                    try:
                        self.dz_restore.C = c2w[:3, 3].detach().clone()
                        self.dz_restore.R = c2w[:3, :3].detach().clone()
                        self.dz_restore.fov_deg = fov_deg
                        self.dz_restore.roll_deg = float(self.roll_slider.value)

                        z_center_model = (
                            result["target"]["pc_depth"][0, 0, h // 2, w // 2, 0]
                            .detach()
                            .float()
                            .item()
                        )
                        z_center = z_center_model / (numerical_scale + 1e-12)

                        self.dz_anchor.Z0 = z_center
                        self.dz_anchor.C0 = self.dz_restore.C.detach().clone()
                        self.dz_anchor.R0 = self.dz_restore.R.detach().clone()
                        self.dz_anchor.fx0 = float(new_fx)
                        self.dz_anchor.fy0 = float(new_fy)
                        self.dz_initialized = True
                        self.dz_state_text.value = "ON"
                        print(
                            f"[DZ] Anchor captured: Z0={self.dz_anchor.Z0:.4f}, fx0={self.dz_anchor.fx0:.2f}"
                        )
                    except Exception as e:
                        print(f"[DZ] Anchor capture failed: {e}")

                img = self._result_to_numpy(result, use_center_dot=True)
                return img

        self.viewer = nerfview.Viewer(
            server=server, render_fn=render_fn, mode="rendering"
        )
        self.show_point_cloud_image_checkbox.on_update(self.viewer.rerender)
        self.show_depth_image_checkbox.on_update(self.viewer.rerender)
        self.show_plucker_moment_checkbox.on_update(self.viewer.rerender)

        def _on_scale_update(handle=None):
            self._refresh_frusta_from_batch()
            if self.viewer is not None:
                self.viewer.rerender(handle)

        self.global_scene_scale.on_update(_on_scale_update)
        self.fov_slider.on_update(self.viewer.rerender)
        self.aspect_ratio_slider.on_update(self.viewer.rerender)
        self.roll_slider.on_update(self.viewer.rerender)
        self.numerical_scale_slider.on_update(self.viewer.rerender)
        self.render_center_red_dot_checkbox.on_update(self.viewer.rerender)

        self._refresh_frusta_from_batch()

        @server.on_client_connect
        def on_client_connect(client: viser.ClientHandle):
            self._clients.add(client)
            try:
                c2w0 = self.batch["c2w"][0, 0].detach().cpu().numpy()
                self._update_viewer_camera_to(c2w0)
            except Exception as e:
                print(f"[connect init camera skipped] {e}")
            print("client connected")

        @server.on_client_disconnect
        def on_client_disconnect(client: viser.ClientHandle):
            if client in self._clients:
                self._clients.remove(client)
            print("client disconnected")

    # ----------------------
    # Frusta & viewer camera helpers
    # ----------------------
    def _refresh_frusta_from_batch(self):
        if not hasattr(self, "server"):
            return
        with self.batch_lock:
            N = int(self.batch["c2w"].shape[1])
            base_c2w = [
                self.batch["c2w"][0, i].detach().cpu().numpy() for i in range(N)
            ]
            T = self.world_T_np
            c2w_list = [T @ c for c in base_c2w]

        scale = float(self.global_scene_scale.value)

        to_remove = [i for i in list(self.frusta.keys()) if i >= N]
        for i in to_remove:
            try:
                self.frusta[i].remove()
            except Exception:
                pass
            self.frusta.pop(i, None)

        for i in range(N):
            c2w_i = c2w_list[i]
            wxyz_xyz = SE3.from_matrix(c2w_i).wxyz_xyz

            if i in self.frusta:
                h = self.frusta[i]
                h.wxyz = wxyz_xyz[..., :4]
                h.position = wxyz_xyz[..., 4:] * scale
            else:
                h = self.server.scene.add_camera_frustum(
                    name=f"camera_{i}",
                    fov=0.5,
                    aspect=1.0,
                    scale=0.1,
                    color=(0.0, 0.0, 1.0),
                    wxyz=wxyz_xyz[..., :4],
                    position=wxyz_xyz[..., 4:] * scale,
                )

                def on_click(e, index=i):
                    print(f"clicked on camera {index}")
                    with self.batch_lock:
                        c2w_now = self.batch["c2w"][0, index].detach().cpu().numpy()
                        c2w_now = self.world_T_np @ c2w_now
                    wxyz_xyz_now = SE3.from_matrix(c2w_now).wxyz_xyz
                    e.client.camera.wxyz = wxyz_xyz_now[..., :4]
                    e.client.camera.position = (
                        wxyz_xyz_now[..., 4:] * self.global_scene_scale.value
                    )

                h.on_click(on_click)
                self.frusta[i] = h

    def _update_viewer_camera_to(self, c2w_np):
        c2w_np = self.world_T_np @ c2w_np
        wxyz_xyz = SE3.from_matrix(c2w_np).wxyz_xyz
        for client in list(self._clients):
            try:
                client.camera.wxyz = wxyz_xyz[..., :4]
                client.camera.position = (
                    wxyz_xyz[..., 4:] * self.global_scene_scale.value
                )
            except Exception:
                pass


@hydra.main(
    version_base=None, config_path="./configs", config_name="demo_viser_from_dataset"
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    config.training.target_has_input = False
    config.training.dynamic_input_view_num = False
    config.training.num_input_views = config.training.get("num_input_views", 5)
    config.training.num_target_views = 1
    config.training.num_views = config.training.num_input_views + 1
    config.inference.if_inference = True
    config.inferece_out_dir = "./experiments/test"
    print(config)

    viser_demo = ViserDemo(config)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
