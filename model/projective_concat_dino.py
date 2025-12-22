import os
import traceback

import einx
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from easydict import EasyDict as edict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time

from utils import camera_utils, data_utils

from .loss import LossComputer
from .transformer import QK_Norm_TransformerBlock, RotaryEmbedding, init_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)
        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()

        # Initialize transformer blocks
        self._init_transformer()

        # Initialize loss computer
        if not config.get("inference", {}).get("if_inference", False):
            self.loss_computer = LossComputer(config)
        else:
            self.loss_computer = None

    def _init_dino(self):
        from transformers import AutoModel

        self.dino = AutoModel.from_pretrained(
            self.config.model.dino.model_path, local_files_only=True
        )
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False

    def _create_tokenizer(self, tokenizer_cfg):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=tokenizer_cfg.patch_size,
                pw=tokenizer_cfg.patch_size,
            ),
            nn.Linear(
                tokenizer_cfg.in_channels * (tokenizer_cfg.patch_size**2),
                self.config.model.transformer.d,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image token decoder (decode image tokens into pixels)
        self._init_dino()
        try:
            with_confidence = self.config.model.image_token_decoder.with_confidence
        except:
            with_confidence = False

        self.image_tokenizer = self._create_tokenizer(self.config.model.image_tokenizer)
        self.target_pose_tokenizer = self._create_tokenizer(
            self.config.model.target_pose_tokenizer
        )

        out_channels = 3 if not with_confidence else 4
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * out_channels,
                bias=False,
            ),
            # nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)

    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        emb_type = self.config.model.get("embedding", None)
        rotary_emb = RotaryEmbedding(config.d_head) if emb_type == "rope" else None
        self.rotary_emb = rotary_emb

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm, rotary_emb=rotary_emb
            )
            for _ in range(config.n_layer)
        ]

        # Apply special initialization if configured
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)

        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)

    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()

    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.

        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.

        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(self.transformer_blocks)

        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens

        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens

        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False,
            )

        return input_tokens

    def get_posed_input(
        self, images=None, ray_o=None, ray_d=None, method="default_plucker"
    ):
        """
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        """

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)

        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)

        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    @torch.no_grad()
    def compose_input_tensor(self, batch, config):
        b, v = batch.c2w.shape[:2]
        h, w = batch.image_h_w
        ret = torch.zeros((b, v, 0, h, w)).to(batch.c2w)

        if config.use_orig_image:
            ret = torch.cat([ret, batch.image], dim=2)
        if config.use_pc_image:
            ret = torch.cat([ret, batch.pc_image], dim=2)
        if config.use_plucker:
            ret = self.get_posed_input(images=ret, ray_o=batch.ray_o, ray_d=batch.ray_d)
        assert (
            ret.shape[2] == config.in_channels
        ), f"Input shape mismatch: required {config.in_channels} channels, input tensor shape {ret.shape}"
        return ret

    def _render_pc_image_for_views(self, input_batch, target_c2w, target_fxfycxcy, height, width):
        """Render point cloud images for arbitrary target views using the training point cloud."""

        if self.process_data.representation not in ["point-cloud-image", "both"]:
            return None
        if getattr(self.process_data, "pc_from", "depth") != "depth":
            return None
        if not hasattr(input_batch, "depth"):
            return None

        device = input_batch.image.device
        masks = input_batch.get(
            "depth_mask",
            torch.ones_like(input_batch.depth, dtype=torch.bool, device=device),
        )

        outputs = self.process_data._render_pointcloud_images_batch(
            images_in=input_batch.image,
            depths_in=input_batch.depth,
            masks_in=masks,
            fxfycxcy_in=input_batch.fxfycxcy,
            c2w_in=input_batch.c2w,
            fxfycxcy_tgt=target_fxfycxcy,
            c2w_tgt=target_c2w,
            H_out=height,
            W_out=width,
            device=device,
            noise=input_batch.get("noise", None),
            render_depths=False,
        )
        pc_image = einx.rearrange("... h w c -> ... c h w", outputs["colors"])
        return pc_image

    def query_dino(self, images, keep_all_tokens=True):
        """
        Query the DINO model for the features of the images.

        Args:
            images: [b, v, c, h, w]
            keep_all_tokens: bool, default True
                Whether to keep all tokens or remove the CLS and register tokens.
        """
        b, v, c, h, w = images.size()
        images = TF.normalize(images.reshape(-1, c, h, w), IMAGENET_MEAN, IMAGENET_STD)
        feat = self.dino(pixel_values=images).last_hidden_state.view(
            b, v, -1, self.config.model.transformer.d
        )  # [b, v, 1 + n_register_tokens + n_patches, d]
        if keep_all_tokens:
            return feat
        return feat[..., 1 + self.config.model.dino.num_register_tokens :, :]

    def query_tokenizer(self, images, tokenizer):
        """
        Query the tokenizer for the features of the images.

        Args:
            images: [b, v, c, h, w]
            tokenizer: Tokenizer to use
            keep_all_tokens: bool, default True. Only for DINO.

        Returns:
            [b, v, n_patches, d]
        """
        b, v, c, h, w = images.size()
        return tokenizer(images).view(b, v, -1, self.config.model.transformer.d)

    def forward(self, data_batch, has_target_image=True, compute_pc_image=False):

        start_time = time.time()
        input, target = self.process_data(
            data_batch,
            has_target_image=has_target_image,
            target_has_input=self.config.training.target_has_input,
            compute_rays=True,
            compute_pc_image=compute_pc_image,
        )

        torch.cuda.synchronize()
        process_data_time = time.time()
        # Process input images
        # @note compose input / target tensors
        posed_input_images = self.compose_input_tensor(
            input, self.config.model.image_tokenizer
        )
        b, v_input, c, h, w = posed_input_images.size()

        keep_all_tokens = self.config.model.dino.keep_all_tokens

        input_img_tokens = self.query_tokenizer(
            posed_input_images, self.image_tokenizer
        )  # [b, v, n_input_tokens, d]

        input_dino_tokens = self.query_dino(posed_input_images, keep_all_tokens)

        _, _, n_input_tokens, d = input_img_tokens.size()  # [b, v, n_input_tokens, d]
        _, _, n_dino_tokens, d = input_dino_tokens.size()
        input_tokens = einx.rearrange(
            "b v n_in d, b v n_dino d -> b (v (n_in + n_dino)) d",
            input_img_tokens,
            input_dino_tokens,
        )
        n_input_tokens += n_dino_tokens

        target_pose_cond = self.compose_input_tensor(
            target, self.config.model.target_pose_tokenizer
        )

        target_pose_tokens = self.query_tokenizer(
            target_pose_cond, self.target_pose_tokenizer
        )
        b, v_target, n_target_tokens, _ = target_pose_tokens.size()
        target_pose_tokens = target_pose_tokens.view(b * v_target, n_target_tokens, d)

        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_tokens,
            "b np d -> (b v_target) np d",
            v_target=v_target,
            np=n_input_tokens * v_input,
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat(
            (repeated_input_img_tokens, target_pose_tokens), dim=1
        )
        # [b * v_target, v_input*n_input_tokens+n_target_tokens, d]

        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(
            concat_img_tokens,
            gradient_checkpoint=True,
            checkpoint_every=checkpoint_every,
        )

        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_input_tokens, n_target_tokens], dim=1
        )  # [b * v_target, v_input*n_input_tokens+n_target_tokens, d], [b * v_target, n_target_tokens, d]

        # [b*v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(
            target_image_tokens[:, -n_target_tokens:, :]
        )

        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        try:
            with_confidence = self.config.model.image_token_decoder.with_confidence
        except:
            with_confidence = False
        rendered_images = rearrange(
            rendered_images,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size,
            w=width // patch_size,
            p1=patch_size,
            p2=patch_size,
            c=4 if with_confidence else 3,
        )
        conf = None
        if with_confidence:
            rendered_images, conf = (
                rendered_images[..., :3, :, :],
                rendered_images[..., 3:, :, :],
            )
            conf = conf.exp() + 1.0
        rendered_images = rendered_images.sigmoid()

        torch.cuda.synchronize()
        forward_time = time.time()

        if has_target_image:
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image,
                conf=conf,
                valid_mask=target.get("valid_mask", None),
            )
        else:
            loss_metrics = None

        runtime_metrics = {
            "process_data_time": process_data_time - start_time,
            "forward_time": forward_time - process_data_time,
        }

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            conf=conf,
            runtime_metrics=runtime_metrics,
        )

        return result

    @torch.no_grad()
    def render_video(
        self,
        data_batch,
        traj_type="interpolate",
        num_frames=60,
        loop_video=False,
        order_poses=False,
    ):
        """
        Render a video from the model.

        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses

        Returns:
            result: Updated with video rendering
        """

        if data_batch.input is None:
            input, target = self.process_data(
                data_batch,
                has_target_image=False,
                target_has_input=self.config.training.target_has_input,
                compute_rays=True,
            )
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        # Prepare input tokens following the forward pass logic
        posed_images = self.compose_input_tensor(
            input, self.config.model.image_tokenizer
        )
        bs, v_input, _, h, w = posed_images.size()

        keep_all_tokens = self.config.model.dino.keep_all_tokens
        input_img_tokens = self.query_tokenizer(
            posed_images, self.image_tokenizer
        )  # [b, v_input, n_img_tokens, d]
        input_dino_tokens = self.query_dino(posed_images, keep_all_tokens)

        _, _, n_img_tokens, d = input_img_tokens.size()
        _, _, n_dino_tokens, _ = input_dino_tokens.size()
        input_tokens = einx.rearrange(
            "b v n_in d, b v n_dino d -> b (v (n_in + n_dino)) d",
            input_img_tokens,
            input_dino_tokens,
        )
        n_input_tokens = n_img_tokens + n_dino_tokens

        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w  # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy  #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros(
                (c2ws.shape[0], c2ws.shape[1], 3, 3), device=device
            )  # [b, v, 3, 3]
            intrinsics[:, :, 0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :, 1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :, 0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4],
                    intrinsics[b],
                    num_frames,
                    order_poses=order_poses,
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0)  # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0)  # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(
                all_c2ws.shape[0], all_c2ws.shape[1], -1, -1
            )
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros(
                (all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device
            )
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Prepare conditioning tensors for the target trajectory views
        height, width = target.image_h_w
        target_view_batch = edict(
            c2w=all_c2ws,
            fxfycxcy=all_fxfycxcy,
            image_h_w=target.image_h_w,
        )

        target_pose_cfg = self.config.model.target_pose_tokenizer
        if getattr(target_pose_cfg, "use_pc_image", False):
            pc_image = self._render_pc_image_for_views(
                input, all_c2ws, all_fxfycxcy, height, width
            )
            if pc_image is not None:
                target_view_batch.pc_image = pc_image
            else:
                target_view_batch.pc_image = torch.zeros(
                    (bs, all_c2ws.shape[1], 3, height, width),
                    device=input.image.device,
                    dtype=input.image.dtype,
                )

        if getattr(target_pose_cfg, "use_orig_image", False):
            # No ground-truth images for novel views; use zeros to keep channel dimensions consistent
            target_view_batch.image = torch.zeros(
                (bs, all_c2ws.shape[1], 3, height, width),
                device=input.image.device,
                dtype=input.image.dtype,
            )

        target_pose_cond = self.compose_input_tensor(
            target_view_batch, target_pose_cfg
        )

        _, num_views, _, _, _ = target_pose_cond.size()
        target_pose_tokens = self.query_tokenizer(
            target_pose_cond, self.target_pose_tokenizer
        )
        _, _, n_target_tokens, _ = target_pose_tokens.size()
        target_pose_tokens = target_pose_tokens.view(
            bs, num_views * n_target_tokens, d
        )

        view_chunk_size = 4

        video_rendering_list = []
        video_conf_list = []
        with_confidence = False
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            repeated_input_img_tokens = repeat(
                input_tokens.detach(),
                "b np d -> (b chunk) np d",
                chunk=cur_view_chunk_size,
                np=n_input_tokens * v_input,
            )

            start_idx, end_idx = (
                cur_chunk * n_target_tokens,
                (cur_chunk + cur_view_chunk_size) * n_target_tokens,
            )
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            cur_target_pose_tokens = rearrange(
                target_pose_tokens[:, start_idx:end_idx, :],
                "b (v_chunk p) d -> (b v_chunk) p d",
                v_chunk=cur_view_chunk_size,
                p=n_target_tokens,
            )

            cur_concat_input_tokens = torch.cat(
                (
                    repeated_input_img_tokens,
                    cur_target_pose_tokens,
                ),
                dim=1,
            )  # [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = self.transformer_input_layernorm(
                cur_concat_input_tokens
            )

            transformer_output_tokens = self.pass_layers(
                cur_concat_input_tokens, gradient_checkpoint=False
            )

            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_input_tokens, n_target_tokens], dim=1
            )

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            video_rendering = self.image_token_decoder(
                pred_target_image_tokens[:, -n_target_tokens:, :]
            )

            video_rendering = rearrange(
                video_rendering,
                "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size,
                w=width // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=4 if with_confidence else 3,
            )

            if with_confidence:
                cur_conf = video_rendering[..., 3:, :, :]
                video_rendering = video_rendering[..., :3, :, :]
                cur_conf = (cur_conf.exp() + 1.0).cpu()
                video_conf_list.append(cur_conf)

            video_rendering = video_rendering.sigmoid().cpu()

            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering
        if with_confidence and video_conf_list:
            data_batch.video_conf = torch.cat(video_conf_list, dim=1)

        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [
                file_name
                for file_name in os.listdir(load_path)
                if file_name.endswith(".pt")
            ]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [
                os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names
            ]
        else:
            ckpt_paths = [load_path]
        try:
            print(f"Loading checkpoint from {ckpt_paths[-1]}")
            checkpoint = torch.load(
                ckpt_paths[-1], map_location="cpu", weights_only=True
            )
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None

        self.load_state_dict(checkpoint["model"], strict=False)
        return 0
