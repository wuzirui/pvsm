import math
import random
from typing import Union

import cv2
import einx
import imageio
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
from jaxtyping import Float

import gsplat


def to_tensor(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def _warp_affine_rgb_mask(img_rgb: np.ndarray, M2x3: np.ndarray, W: int, H: int):
    mask = np.ones((H, W), np.uint8) * 255
    dst_bgr = cv2.warpAffine(
        img_rgb[..., ::-1].astype(np.float32),
        M2x3.astype(np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_out = cv2.warpAffine(
        mask,
        M2x3.astype(np.float32),
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dst_bgr[..., ::-1], mask_out > 0


def _warp_persp_rgb_mask(img_rgb: np.ndarray, H3x3: np.ndarray, W: int, H: int):
    mask = np.ones((H, W), np.uint8) * 255
    dst_bgr = cv2.warpPerspective(
        img_rgb[..., ::-1].astype(np.float32),
        H3x3.astype(np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_out = cv2.warpPerspective(
        mask,
        H3x3.astype(np.float32),
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dst_bgr[..., ::-1], mask_out > 0


def _fx_from_fovx(fovx_deg: float, W: int) -> float:
    fovx_rad = math.radians(min(max(fovx_deg, 1e-3), 179.9))
    return 0.5 * W / math.tan(0.5 * fovx_rad)


def _tf_roll_rgb_mask(img: np.ndarray, fx, fy, cx, cy, W, H, angle_deg: float):
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    img2, mask = _warp_affine_rgb_mask(img, M, W, H)
    return img2, (fx, fy, cx, cy), mask, M


def _tf_yawpitch_rgb_mask(
    img: np.ndarray, fx, fy, cx, cy, W, H, yaw_deg: float, pitch_deg: float
):
    Ky = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = math.radians(yaw_deg)
    rx = math.radians(pitch_deg)
    Ry = np.array(
        [[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]],
        dtype=np.float32,
    )
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]],
        dtype=np.float32,
    )
    R = Rx @ Ry
    Hm = Ky @ R @ np.linalg.inv(Ky)
    img2, mask = _warp_persp_rgb_mask(img, Hm, W, H)
    return img2, (fx, fy, cx, cy), mask, R, Hm


def _tf_fov_rgb_mask(img: np.ndarray, fx, fy, cx, cy, W, H, fovx_target_deg: float):
    fx_new = _fx_from_fovx(fovx_target_deg, W)
    s = float(fx_new) / max(float(fx), 1e-6)
    T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float32)
    T2 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    M3 = (T1 @ S @ T2).astype(np.float32)
    M = M3[:2, :]
    img2, mask = _warp_affine_rgb_mask(img, M, W, H)
    fy_new = fy * s
    return img2, (fx_new, fy_new, cx, cy), mask, M


def _tf_aniso_rgb_mask(img: np.ndarray, fx, fy, cx, cy, W, H, ratio_fx_over_fy: float):
    r = float(ratio_fx_over_fy)
    sx = math.sqrt(max(r, 1e-8))
    sy = 1.0 / max(sx, 1e-8)
    T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
    A = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float32)
    T2 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    M3 = (T1 @ A @ T2).astype(np.float32)
    M = M3[:2, :]
    img2, mask = _warp_affine_rgb_mask(img, M, W, H)
    fx_new = fx * sx
    fy_new = fy * sy
    return img2, (fx_new, fy_new, cx, cy), mask, M


def _Rz_cam(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [
            [math.cos(a), -math.sin(a), 0.0],
            [math.sin(a), math.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


import torch
from jaxtyping import Float
from typing import Tuple, Optional

def random_mask_image_batch(
    image: Float[torch.Tensor, "b n c h w"],
    patch_drop_rate: float,
    pixel_drop_rate: float,
    patch_size: int,
    patch_keep_rate: float = 0.0,
    color_mul_range: Tuple[float, float] = (0.9, 1.1),
    color_add_range: Tuple[float, float] = (-0.05, 0.05),
    per_channel: bool = True,
    clip: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    1) patch 丢弃；2) 像素丢弃；3) 强制保留部分 patch；4) patch 级颜色扰动（乘 & 加常数）。
    返回被 mask 后的图像（丢弃处置 0）。

    Args:
        image: (B,N,C,H,W)
        patch_drop_rate: [0,1] 每个 patch 的丢弃概率
        pixel_drop_rate: [0,1] 每个像素的丢弃概率
        patch_size: >0，方形 patch 尺寸
        patch_keep_rate: [0,1] 最终强制保留的 patch 占比（覆盖前两步丢弃）
        color_mul_range: 乘性扰动的均匀采样区间
        color_add_range: 加性扰动的均匀采样区间（假设像素已归一化到[0,1]）
        per_channel: True 则每通道独立扰动，否则各通道共用同一扰动
        clip: True 则将结果裁剪到 [0,1]
        generator: 可选随机数发生器（便于可复现）
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    for name, val in [
        ("patch_drop_rate", patch_drop_rate),
        ("pixel_drop_rate", pixel_drop_rate),
        ("patch_keep_rate", patch_keep_rate),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {val}")
    mul_lo, mul_hi = color_mul_range
    add_lo, add_hi = color_add_range
    if mul_lo > mul_hi or add_lo > add_hi:
        raise ValueError("color_*_range low must be <= high")

    B, N, C, H, W = image.shape
    device, dtype = image.device, image.dtype

    # ---- patch 网格（向上取整）----
    gh = (H + patch_size - 1) // patch_size
    gw = (W + patch_size - 1) // patch_size
    P = gh * gw

    # ---- 1) patch-wise keep mask ----
    patch_keep_prob = 1.0 - patch_drop_rate
    patch_keep = (torch.rand((B, N, 1, gh, gw), device=device, generator=generator) < patch_keep_prob).to(dtype)
    patch_mask = patch_keep.repeat_interleave(patch_size, dim=3).repeat_interleave(patch_size, dim=4)
    patch_mask = patch_mask[:, :, :, :H, :W]  # (B,N,1,H,W)

    # ---- 2) pixel-wise keep mask ----
    pixel_keep_prob = 1.0 - pixel_drop_rate
    pixel_keep = (torch.rand((B, N, 1, H, W), device=device, generator=generator) < pixel_keep_prob).to(dtype)

    keep_mask = patch_mask * pixel_keep  # (B,N,1,H,W)

    # ---- 3) 强制保留若干完整 patch ----
    k = int(round(patch_keep_rate * P))
    if k > 0:
        scores = torch.rand((B, N, P), device=device, generator=generator)
        idx = scores.topk(k=min(k, P), dim=-1).indices  # (B,N,k)
        keep_lin = torch.zeros((B, N, P), device=device, dtype=torch.bool)
        keep_lin.scatter_(dim=-1, index=idx, value=True)
        keep_patch = keep_lin.view(B, N, 1, gh, gw).to(dtype)
        keep_patch = keep_patch.repeat_interleave(patch_size, dim=3).repeat_interleave(patch_size, dim=4)
        keep_patch = keep_patch[:, :, :, :H, :W]  # (B,N,1,H,W)
        keep_mask = torch.maximum(keep_mask, keep_patch)

    # ---- 4) patch-level 颜色扰动：乘 + 加 ----
    ch_dim = C if per_channel else 1
    # 采样乘性和加性参数（均匀分布）
    mul = torch.empty((B, N, ch_dim, gh, gw), device=device).uniform_(mul_lo, mul_hi, generator=generator)
    add = torch.empty((B, N, ch_dim, gh, gw), device=device).uniform_(add_lo, add_hi, generator=generator)

    # 广播到像素级并裁剪大小
    mul = mul.repeat_interleave(patch_size, dim=3).repeat_interleave(patch_size, dim=4)[:, :, :, :H, :W]
    add = add.repeat_interleave(patch_size, dim=3).repeat_interleave(patch_size, dim=4)[:, :, :, :H, :W]

    # 若 per_channel=False，需要在通道维上广播到 C
    if not per_channel:
        mul = mul.expand(B, N, C, H, W)
        add = add.expand(B, N, C, H, W)

    # 应用颜色扰动（先扰动，再用 keep_mask 置零丢弃处）
    jittered = image * mul.to(dtype) + add.to(dtype)

    if clip:
        jittered = jittered.clamp(0.0, 1.0)

    # 最终应用 mask
    return jittered * keep_mask


def unproject_depth_map_to_point_map_batch(
    depth_map: torch.Tensor, c2w: torch.Tensor, fxfycxcy: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (..., 1, H, W) or (..., H, W)
        c2w (np.ndarray): Batch of camera extrinsic matrices of shape (..., 3, 4) or (..., 4, 4)
        fxfycxcy (np.ndarray): Batch of camera intrinsic matrices of shape (..., 4)

    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (..., H, W, 3)
    """
    depth_map = to_tensor(depth_map)
    c2w = to_tensor(c2w)
    fxfycxcy = to_tensor(fxfycxcy)
    if einx.matches("... p q", c2w, p=4, q=4):
        c2w = c2w[..., :3, :]
    prefix = c2w.shape[:-2]

    depth_map = einx.rearrange("... h w-> (...) h w", depth_map)
    c2w = einx.rearrange("... p q -> (...) p q", c2w)
    fxfycxcy = einx.rearrange("... r -> (...) r", fxfycxcy)
    b, h, w = depth_map.shape

    cam_coords_points = depth_to_cam_coords_points_batch(depth_map, fxfycxcy, w, h)
    R_cam_to_world = c2w[:, :3, :3]
    t_cam_to_world = c2w[:, :3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = einx.add(
        "b h w i, b i -> b h w i",
        einx.dot("b i j, b h w j -> b h w i", R_cam_to_world, cam_coords_points),
        t_cam_to_world,
    )  # HxWx3, 3x3 -> HxWx3

    return world_coords_points.reshape(prefix + (h, w, 3))


def depth_to_cam_coords_points_batch(
    depth_map: Float[torch.Tensor, "b h w"],
    fxfycxcy: Float[torch.Tensor, "b 4"],
    W: int,
    H: int,
) -> tuple[Float[torch.Tensor, "b h w 3"], Float[torch.Tensor, "b h w 3"]]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    # Convert depth map to camera coordinates
    # cam_coords_points = depth_to_cam_coords_points_batch(depth_map, fxfycxcy)
    fu, fv, cu, cv = fxfycxcy.split(1, dim=-1)

    # Generate grid of pixel coordinates
    v, u = torch.meshgrid(
        torch.arange(H, device=depth_map.device),
        torch.arange(W, device=depth_map.device),
        indexing="ij",
    )

    # Unproject to camera coordinates
    # x_cam = (u - cu) * depth_map / fu
    x_cam = einx.divide(
        "b h w, b 1 -> b h w", einx.subtract("h w, b 1 -> b h w", u + 0.5, cu) * depth_map, fu
    )
    # y_cam = (v - cv) * depth_map / fv
    y_cam = einx.divide(
        "b h w, b 1 -> b h w", einx.subtract("h w, b 1 -> b h w", v + 0.5, cv) * depth_map, fv
    )
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), axis=-1)

    return cam_coords


def gaussian_from_points_batch(
    points: Float[torch.Tensor, "b n_cams h w 3"],
    colors: Float[torch.Tensor, "b n_cams 3 h w"],
    point_masks: Float[torch.Tensor, "b n_cams 1 h w"],
    scale: float,
    opacity: float,
) -> dict[str, Float[torch.Tensor, "b n_points c"]]:
    """
    Create gaussian from points.

    Args:
        points: Points to create gaussian from.
        colors: Colors of the points.
        point_masks: Masks of the points.
        scale: Scale of the gaussian.
        opacity: Opacity of the gaussian.

    Returns:
        A dictionary of gaussians.
    """
    quats = torch.zeros(
        (*points.shape[:-1], 4), device=points.device, dtype=points.dtype
    )
    quats[..., 0] = 1
    scales = torch.ones_like(points) * scale
    opacities = (point_masks * opacity).to(points)
    return {
        "means": einx.rearrange("b n_cams h w p -> b (n_cams h w) p", points, p=3),
        "quats": einx.rearrange("b n_cams h w q -> b (n_cams h w) q", quats, q=4),
        "scales": einx.rearrange("b n_cams h w s -> b (n_cams h w) s", scales, s=3),
        "opacities": einx.rearrange("b n_cams 1 h w -> b (n_cams h w) 1", opacities),
        "colors": einx.rearrange("b n_cams c h w -> b (n_cams h w) c", colors, c=3).to(
            points
        ),
    }


def concat_gaussians(gaussians_1, gaussians_2):
    return {
        "means": torch.cat([gaussians_1["means"], gaussians_2["means"]], dim=-2),
        "quats": torch.cat([gaussians_1["quats"], gaussians_2["quats"]], dim=-2),
        "scales": torch.cat([gaussians_1["scales"], gaussians_2["scales"]], dim=-2),
        "opacities": torch.cat(
            [gaussians_1["opacities"], gaussians_2["opacities"]], dim=-2
        ),
        "colors": torch.cat([gaussians_1["colors"], gaussians_2["colors"]], dim=-2),
    }


def build_K_from_fxfycxcy(
    fxfycxcy: Float[torch.Tensor, "b n_cams 4"],
) -> Float[torch.Tensor, "b n_cams 3 3"]:
    b, n_cams, _ = fxfycxcy.shape
    ret = torch.zeros((b, n_cams, 3, 3), device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    ret[..., 0, 0] = fxfycxcy[..., 0]
    ret[..., 1, 1] = fxfycxcy[..., 1]
    ret[..., 0, 2] = fxfycxcy[..., 2]
    ret[..., 1, 2] = fxfycxcy[..., 3]
    ret[..., 2, 2] = 1
    return ret


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def render_pointcloud_images_batch(
    gaussians: dict[str, Float[torch.Tensor, "b n_points c"]],
    c2w: Float[torch.Tensor, "b n_cams 4 4"],
    fxfycxcy: Float[torch.Tensor, "b n_cams 4"],
    width: int,
    height: int,
    render_depths: bool = False,
    near: float = 0.1,
    sh_degree: int = None,
):
    """
    Render pointcloud images.

    Args:
        gaussians: Gaussians to render.
        c2w: Camera to world matrices.
        fxfycxcy: Camera intrinsic parameters.
        width: Width of the image.
        height: Height of the image.
    Returns:
        colors: (B,V_t,H_out,W_out,3)
        alphas: (B,V_t,H_out,W_out,1)
    """
    b, n_cams, _, _ = c2w.shape
    viewmats = closed_form_inverse_se3(c2w.reshape(-1, 4, 4)).reshape(b, n_cams, 4, 4)
    Ks = build_K_from_fxfycxcy(fxfycxcy)

    colors, alphas, meta = gsplat.rasterization(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=gaussians["scales"],
        opacities=gaussians["opacities"][..., 0],
        colors=gaussians["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB+ED" if render_depths else "RGB",
        near_plane=near,
        far_plane=1e10,
        sh_degree=sh_degree,
    )
    outputs = {
        "colors": colors[..., :3],
        "alphas": alphas,
        "meta": meta,
    }
    if render_depths:
        outputs["depths"] = colors[..., 3:]
    return outputs


def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)


def to_uint8_img(img_float01: torch.Tensor):
    return (img_float01.clamp(0, 1) * 255).byte().cpu().numpy()


# ---------------- Geometry ----------------


@torch.no_grad()
def make_sphere_shell(
    B: int,  # batch size
    N_pts: int,  # 球壳点数
    radius: float,  # 半径 R (世界坐标)
    device,
    dtype=torch.float32,
    center: torch.Tensor | None = None,  # (B,3) 或 None
    scale_mul: float = 0.5,  # 高斯尺度系数，最终 scale = scale_mul * spacing
    opacity: float = 0.7,  # 球壳点的不透明度
    per_batch_random_color: bool = False,  # 是否每个 batch 重新采样颜色
    jitter: float = 0.0,  # 球面法向方向的轻微抖动比例 (0~0.1)
):
    """
    生成批量球壳高斯参数（均匀分布 Fibonacci sphere）。
    返回:
      means_bg:   (B, N_pts, 3)
      colors_bg:  (B, N_pts, 3) in [0,1]
      quats_bg:   (B, N_pts, 4)  wxyz, 全单位旋转
      scales_bg:  (B, N_pts, 3)  各向同性
      opac_bg:    (B, N_pts, 1)
    """
    # --- 球面均匀采样（Fibonacci sphere） ---
    # k ∈ [0,N-1], y=1-2*(k+0.5)/N, phi = golden_angle*k
    N = int(N_pts)
    k = torch.arange(N, device=device, dtype=dtype) + 0.5
    y = 1.0 - 2.0 * (k / float(N))  # [-1,1]
    r_xy = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))  # 圆半径
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden_angle * k

    x = r_xy * torch.cos(phi)
    z = r_xy * torch.sin(phi)
    # 单位球坐标
    dirs = torch.stack([x, y, z], dim=-1)  # (N,3)

    if jitter > 0:
        # 沿法向的轻微噪声，避免 pattern 痕迹
        noise = (torch.rand_like(dirs) - 0.5) * 2.0 * jitter
        dirs = torch.nn.functional.normalize(dirs + noise, dim=-1)

    # 半径 R
    base = dirs * float(radius)  # (N,3)

    # 如果提供 center，则各样本可有不同中心
    if center is None:
        center = torch.zeros(B, 3, device=device, dtype=dtype)
    else:
        assert center.shape == (B, 3)

    means_bg = base.unsqueeze(0) + center.unsqueeze(1)  # (B,N,3)

    # --- 颜色 ---
    if per_batch_random_color:
        colors_bg = torch.rand(B, N, 3, device=device, dtype=dtype)
    else:
        colors_1 = torch.rand(1, N, 3, device=device, dtype=dtype)
        colors_bg = colors_1.expand(B, -1, -1)

    # --- 高斯尺度：由点间距估算 ---
    # 平均面积/点 A = 4πR^2 / N，邻接间距 ~ sqrt(A)
    spacing = 2.0 * float(radius) * np.sqrt(np.pi / float(N))  # s ≈ 2R*sqrt(pi/N)
    scale_val = float(scale_mul) * float(spacing)
    scales_bg = torch.full((B, N, 3), scale_val, device=device, dtype=dtype)

    # --- 姿态/不透明度 ---
    quats_bg = torch.zeros(B, N, 4, device=device, dtype=dtype)
    quats_bg[..., 0] = 1.0  # wxyz 单位旋转
    opac_bg = torch.full((B, N, 1), float(opacity), device=device, dtype=dtype)

    return means_bg, colors_bg.clamp(0, 1), quats_bg, scales_bg, opac_bg


class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.representation = config.get("data_process", {}).get(
            "representation", "plucker-ray"
        )
        assert self.representation in [
            "point-cloud-image",
            "plucker-ray",
            "both",
        ], f"Invalid representation: {self.representation}"

        pc_cfg = getattr(config.data_process, "point_cloud", edict())
        self.pc_from = getattr(pc_cfg, "from", "depth")
        assert self.pc_from in ["depth", "random-mask"], f"Invalid pc_from: {self.pc_from}"
        self.pc_scale = float(getattr(pc_cfg, "gauss_scale", 0.005))
        self.pc_opacity = float(getattr(pc_cfg, "opacity", 1.0))
        self.pc_eps2d = float(getattr(pc_cfg, "eps2d", 0.3))
        self.pc_radius_clip = float(getattr(pc_cfg, "radius_clip", 0.0))
        self.pc_patch_drop_rate = float(getattr(pc_cfg, "patch_drop_rate", 0.9))
        self.pc_pixel_drop_rate = float(getattr(pc_cfg, "pixel_drop_rate", 0.2))
        self.pc_patch_keep_rate = float(getattr(pc_cfg, "patch_keep_rate", 0.05))
        self.compute_input_pc_image = getattr(
            pc_cfg, "compute_input_pc_image", False
        )
        self.override_input_image = getattr(pc_cfg, "override_input_image", False)

        tp_cfg = getattr(config.data_process, "target_perturbation", None)

        self.target_perturb = None
        if tp_cfg and getattr(tp_cfg, "enable", False):
            modes = list(getattr(tp_cfg, "modes", ["roll", "yawpitch", "fov", "anisopixel"]))
            probability = float(getattr(tp_cfg, "probability", 1.0))
            probability = min(max(probability, 0.0), 1.0)
            roll_choices = list(
                getattr(
                    tp_cfg,
                    "roll_angle_choices",
                    [-55.0, -35.0, -15.0, 15.0, 35.0, 55.0],
                )
            )
            yaw_range = tuple(getattr(tp_cfg, "yaw_range", (-25.0, 25.0)))
            pitch_range = tuple(getattr(tp_cfg, "pitch_range", (-15.0, 15.0)))
            fov_range = tuple(getattr(tp_cfg, "fov_range", (60.0, 120.0)))
            aniso_choices = list(
                getattr(tp_cfg, "anisopixel_choices", [0.8, 0.9, 1.1, 1.25])
            )
            if modes:
                self.target_perturb = edict(
                    enable=tp_cfg.enable,
                    modes=modes,
                    probability=probability,
                    roll_angle_choices=roll_choices,
                    yaw_range=yaw_range,
                    pitch_range=pitch_range,
                    fov_range=fov_range,
                    anisopixel_choices=aniso_choices,
                )

    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'depth' (torch.Tensor): Shape [b, v, 1, h, w], optional for some target views
                - 'depth_mask' (torch.Tensor): Shape [b, v, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if self.config.training.get("dynamic_input_view_num", False):
            num_input_views = np.random.randint(
                self.config.training.min_input_views,
                self.config.training.max_input_views + 1,
            )
            num_target_views = self.config.training.num_views - num_input_views
        else:
            assert isinstance(
                self.config.training.num_input_views, int
            ), "num_input_views must be an integer"
            num_input_views = self.config.training.num_input_views
            num_target_views = self.config.training.num_target_views

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views

        num_views, bs = data_batch["c2w"].size(1), data_batch["image"].size(0)
        assert (
            num_target_views < num_views
        ), f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."

        # we here just use first num_input_views as input views and the rest as target views
        # the random part should be done in the dataset class

        # Decide the target view indices
        # if target_has_input and not self.config.training.get("use_index_file", False):
        #     # Randomly sample target views across all views
        #     index = torch.tensor(
        #         [random.sample(range(num_views), num_target_views) for _ in range(bs)],
        #         dtype=torch.long,
        #         device=data_batch["image"].device,
        #     )  # [b, num_target_views]
        # else:
        assert (
            num_input_views + num_target_views <= self.config.training.num_views
        ), f"We have {self.config.training.num_views} views in total, but we want to select {num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."

        index = torch.tensor(
            [[num_views - 1 - j for j in range(num_target_views)] for _ in range(bs)],
            dtype=torch.long,
            device=data_batch["image"].device,
        )
        index = torch.sort(index, dim=1).values  # [b, num_target_views]

        for key, value in data_batch.items():
            if key == "scene_name":
                input_dict[key] = value
                target_dict[key] = value
                continue
            input_dict[key] = value[:, :num_input_views, ...]

            to_expand_dim = value.shape[
                2:
            ]  # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(
                index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)
            ).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision
            if key == "image" and not has_target_image:
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)

        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        target_dict = self._apply_target_perturbation(target_dict, has_target_image)

        return input_dict, target_dict

    def _apply_target_perturbation(self, target_dict, has_target_image: bool):
        """Apply random camera perturbations to target views if enabled."""
        if self.target_perturb is None:
            return target_dict
        height, width = target_dict["image_h_w"]
        Ks = target_dict.get("fxfycxcy")
        if Ks is None:
            return target_dict

        B, V = Ks.shape[:2]
        device = Ks.device
        image_tensor = target_dict.get("image") if has_target_image else None
        mask_dtype = image_tensor.dtype if image_tensor is not None else Ks.dtype
        masks = torch.ones((B, V, 1, height, width), dtype=mask_dtype, device=device)

        if (
            self.target_perturb is None
            or image_tensor is None
            or not has_target_image
            or image_tensor.numel() == 0
        ):
            target_dict["valid_mask"] = masks
            return target_dict

        c2w = target_dict.get("c2w")
        assert c2w is not None, "Target dictionary must contain c2w for perturbations."

        for b in range(B):
            for v in range(V):
                if random.random() > self.target_perturb.probability:
                    continue

                mode = random.choice(self.target_perturb.modes)

                img_np = (
                    image_tensor[b, v]
                    .detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                    .astype(np.float32)
                )
                fx, fy, cx, cy = [
                    float(x)
                    for x in Ks[b, v]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                ]
                c2w_np = c2w[b, v].detach().cpu().numpy().astype(np.float32)
                H, W = img_np.shape[:2]

                updated = False

                if mode == "roll":
                    if not self.target_perturb.roll_angle_choices:
                        continue
                    angle = float(random.choice(self.target_perturb.roll_angle_choices))
                    img_np2, intr_new, mask_np, _ = _tf_roll_rgb_mask(
                        img_np, fx, fy, cx, cy, W, H, angle
                    )
                    R_cam = _Rz_cam(angle)
                    c2w_np[:3, :3] = c2w_np[:3, :3] @ R_cam
                    updated = True
                elif mode == "yawpitch":
                    yaw_min, yaw_max = self.target_perturb.yaw_range
                    pitch_min, pitch_max = self.target_perturb.pitch_range
                    yaw_deg = random.uniform(float(yaw_min), float(yaw_max))
                    pitch_deg = random.uniform(float(pitch_min), float(pitch_max))
                    img_np2, intr_new, mask_np, R_cam, _ = _tf_yawpitch_rgb_mask(
                        img_np, fx, fy, cx, cy, W, H, yaw_deg, pitch_deg
                    )
                    c2w_np[:3, :3] = c2w_np[:3, :3] @ R_cam.T
                    updated = True
                elif mode == "fov":
                    fov_min, fov_max = self.target_perturb.fov_range
                    fov_deg = random.uniform(float(fov_min), float(fov_max))
                    img_np2, intr_new, mask_np, _ = _tf_fov_rgb_mask(
                        img_np, fx, fy, cx, cy, W, H, fov_deg
                    )
                    updated = True
                elif mode == "anisopixel":
                    if not self.target_perturb.anisopixel_choices:
                        continue
                    ratio = float(random.choice(self.target_perturb.anisopixel_choices))
                    img_np2, intr_new, mask_np, _ = _tf_aniso_rgb_mask(
                        img_np, fx, fy, cx, cy, W, H, ratio
                    )
                    updated = True
                else:
                    continue

                if not updated:
                    continue

                img_np2 = np.clip(img_np2, 0.0, 1.0)
                mask_np = mask_np.astype(np.float32)

                image_tensor[b, v] = (
                    torch.from_numpy(img_np2)
                    .permute(2, 0, 1)
                    .to(image_tensor.device, dtype=image_tensor.dtype)
                )
                Ks[b, v] = torch.tensor(
                    intr_new, dtype=Ks.dtype, device=Ks.device
                )
                c2w[b, v] = torch.from_numpy(c2w_np).to(c2w.device, dtype=c2w.dtype)
                masks[b, v] = (
                    torch.from_numpy(mask_np)
                    .unsqueeze(0)
                    .to(masks.device, dtype=masks.dtype)
                )

        target_dict["valid_mask"] = masks
        return target_dict

    # ---------------- batch gsplat rasterization ----------------
    def _get_backside_color(self, color_mode, images_in):
        if color_mode == "single_random":
            return einx.rearrange(
                "b c -> b n c h w",
                torch.rand(images_in.shape[0], 3),
                n=images_in.shape[1],
                c=3,
                h=images_in.shape[3],
                w=images_in.shape[4],
            )
        elif color_mode == "all_pixel_random":
            return torch.rand_like(images_in)
        elif color_mode == "grey":
            return torch.ones_like(images_in) * 0.5
        else:
            raise ValueError(f"Invalid color mode: {color_mode}")

    @torch.no_grad()
    def _render_pointcloud_images_batch(
        self,
        images_in,
        depths_in,
        masks_in,
        fxfycxcy_in,
        c2w_in,  # (B,V_in, ...)
        fxfycxcy_tgt,
        c2w_tgt,  # (B,V_t, ...)
        H_out,
        W_out,
        device,
        render_depths: bool = False,
        noise: torch.Tensor = None,
        near: float = 0.1,
    ):
        points_world = unproject_depth_map_to_point_map_batch(
            depths_in, c2w_in, fxfycxcy_in
        )
        gaussians = gaussian_from_points_batch(
            points_world, images_in, masks_in, self.pc_scale, self.pc_opacity
        )

        shell_cfg = self.config.data_process.point_cloud.get("bg_shell", edict())
        if shell_cfg.get("enable", False):
            gaussians = concat_gaussians(
                gaussians,
                self._make_shell(
                    shell_cfg, images_in.shape[0], images_in.dtype, device
                ),
            )

        backside = self.config.data_process.point_cloud.get("backside", None)
        if backside and backside.get("enable", False):
            delta_depth = self.config.data_process.point_cloud.backside.get(
                "delta_depth", 1e-3
            )
            back_opacity = self.config.data_process.point_cloud.backside.get(
                "opacity", 0.3
            )
            points_backside = unproject_depth_map_to_point_map_batch(
                depths_in + delta_depth, c2w_in, fxfycxcy_in
            )
            backside_color = self.config.data_process.point_cloud.backside.get(
                "color", "single_random"
            )
            if noise is None:
                # use random noise if not given
                noise = self._get_backside_color(backside_color, images_in)
            gaussians_backside = gaussian_from_points_batch(
                points_backside,
                noise,
                masks_in,
                self.pc_scale,
                back_opacity,
            )
            gaussians = concat_gaussians(gaussians, gaussians_backside)

        outputs = render_pointcloud_images_batch(
            gaussians, c2w_tgt, fxfycxcy_tgt, W_out, H_out, render_depths=render_depths, near=near
        )
        return outputs

    # ---------------- forward ----------------
    @torch.no_grad()
    def forward(
        self,
        data_batch,
        has_target_image=True,
        target_has_input=True,
        compute_rays=True,
        compute_pc_image=True,
        return_depths=True,
    ):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.

        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(
            data_batch,
            has_target_image=has_target_image,
            target_has_input=target_has_input,
        )

        if self.representation in ["plucker-ray", "both"] and compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                ray_o, ray_d = compute_rays_batch(
                    c2w,
                    fxfycxcy,
                    image_height,
                    image_width,
                    device=data_batch["image"].device,
                )
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        B = data_batch["image"].shape[0]
        imgH, imgW = data_batch["image"].shape[3], data_batch["image"].shape[4]
        H_out, W_out = imgH, imgW

        device = data_batch["image"].device
        # 点云投影表示（批量渲染，不逐 sample）
        if (self.representation in ["point-cloud-image", "both"] or compute_pc_image) and self.pc_from == "depth":
            # === 单次渲染：拼接 target + input 相机，再按视角数拆分 ===
            V_t = target_dict["fxfycxcy"].shape[1]
            V_in = input_dict["fxfycxcy"].shape[1]

            if self.compute_input_pc_image:
                Ks_all = torch.cat(
                    [target_dict["fxfycxcy"], input_dict["fxfycxcy"]], dim=1
                )  # (B,V_t+V_in,4)
                c2w_all = torch.cat(
                    [target_dict["c2w"], input_dict["c2w"]], dim=1
                )  # (B,V_t+V_in,4,4)
            else:
                Ks_all = target_dict["fxfycxcy"]
                c2w_all = target_dict["c2w"]

            outputs = self._render_pointcloud_images_batch(
                images_in=input_dict["image"],  # (B,V_in,3,H,W)
                depths_in=input_dict["depth"],  # (B,V_in,1,H,W)
                masks_in=input_dict.get(
                    "depth_mask", torch.ones_like(input_dict["depth"]).bool()
                ),
                fxfycxcy_in=input_dict["fxfycxcy"],  # (B,V_in,4)
                c2w_in=input_dict["c2w"],  # (B,V_in,4,4)
                fxfycxcy_tgt=Ks_all,  # (B,V_t{+V_in},4)
                c2w_tgt=c2w_all,  # (B,V_t{+V_in},4,4)
                H_out=H_out,
                W_out=W_out,
                device=device,
                noise=input_dict.get("noise", None),
                render_depths=return_depths,
            )
            rgb_all = einx.rearrange("... h w c -> ... c h w", outputs["colors"])
            if return_depths:
                depth_all = einx.rearrange("... h w -> ... h w", outputs["depths"])
                target_dict["pc_depth"] = depth_all[:, :V_t]

            # 拆分结果：前 V_t -> target，后 V_in -> input（仅当需要时）
            target_dict["pc_image"] = rgb_all[:, :V_t]
            # target_dict["pc_depth"] = depth_all[:, :V_t]  # 若需要深度，解开

            if self.compute_input_pc_image:
                input_pc_image = rgb_all[:, V_t : V_t + V_in]
                # input_pc_depth = depth_all[:, V_t:V_t + V_in]
                input_dict["pc_image"] = input_pc_image  # (B,V_in,3,H_out,W_out)

        if (self.representation in ["point-cloud-image", "both"] or compute_pc_image) and self.pc_from == "random-mask":
            target_dict["pc_image"] = random_mask_image_batch(
                target_dict["image"],
                patch_drop_rate=self.pc_patch_drop_rate,
                pixel_drop_rate=self.pc_pixel_drop_rate,
                patch_keep_rate=self.pc_patch_keep_rate,
                patch_size=self.config.model.image_tokenizer.patch_size // 2,
            )

        return input_dict, target_dict

    def _make_shell(self, bg_cfg, B, dtype, device):
        N_bg = int(getattr(bg_cfg, "num_points", 20000))
        R_bg = float(getattr(bg_cfg, "radius", 100.0))
        op_bg = float(getattr(bg_cfg, "opacity", 0.7))
        sclmul = float(getattr(bg_cfg, "scale_mul", 0.5))
        jitter = float(getattr(bg_cfg, "jitter", 0.0))
        per_batch_random_color = bool(getattr(bg_cfg, "per_batch_random_color", False))

        center = None

        means_bg, colors_bg, quats_bg, scales_bg, opac_bg = make_sphere_shell(
            B=B,
            N_pts=N_bg,
            radius=R_bg,
            device=device,
            dtype=dtype,
            center=center,
            scale_mul=sclmul,
            opacity=op_bg,
            per_batch_random_color=per_batch_random_color,
            jitter=jitter,
        )

        return {
            "means": means_bg,
            "colors": colors_bg,
            "quats": quats_bg,
            "scales": scales_bg,
            "opacities": opac_bg,
        }


@torch.no_grad()
def compute_rays_batch(c2w, fxfycxcy, h=None, w=None, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fx, fy, cx, cy = (
        fxfycxcy[:, :, 0],
        fxfycxcy[:, :, 1],
        fxfycxcy[:, :, 2],
        fxfycxcy[:, :, 3],
    )
    h_orig = int(
        2 * cy.max().item()
    )  # Original height (estimated from the intrinsic matrix)
    w_orig = int(
        2 * cx.max().item()
    )  # Original width (estimated from the intrinsic matrix)
    if h is None or w is None:
        h, w = h_orig, w_orig

    # in case the ray/image map has different resolution than the original image
    if h_orig != h or w_orig != w:
        fx = fx * w / w_orig
        fy = fy * h / h_orig
        cx = cx * w / w_orig
        cy = cy * h / h_orig

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d
