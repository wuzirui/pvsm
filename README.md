<div align="center">
<p align="center">
  <img src="assets/logo.png" alt="PVSM Logo" width="90%"/>
</p>
<p align="center">  
    <a href="https://wuzirui.github.io">Zirui Wu</a>,
    <a href="https://jzr99.github.io/">Zeren Jiang</a>,
    <a href="https://oswaldm.github.io/">Martin R. Oswald</a>,
    <a href="https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page/SONG-Jie/jsongroas">Jie Song</a>

</p>

<div align="center">
    <a href="https://wuzirui.github.io/pvsm-web"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2601.05116"><strong>ArXiV</strong></a> |
    <a href="https://youtu.be/HSQnrXCU_BM"><strong>YouTube</strong></a>
</div>


### CVPR 2026

</div>

## Installation

```bash
conda create -n pvsm python=3.11
conda activate pvsm
# Install torch torchvision based on your environment configurations
pip install -r requirements.txt
```

There's a [known issue](https://github.com/nerfstudio-project/gsplat/pull/772) of the current release of `gsplat==1.5.3`, so please install `gsplat` via source for now:
```bash
# Install gsplat from source
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

## Quick Start

### Download Checkpoints

Download [DINOv3-ViT-B](https://github.com/nerfstudio-project/gsplat/pull/772) and place it under `metric_checkpoints/`;

Download our pre-trained model checkpoints:

1. **12-layer model** (small): [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/zwu797_connect_hkust-gz_edu_cn/IQBXhmE6WhhvQ4mIOtno-XbqAcbfA56ldRWyrFZWqZHKrho?e=pD00NF)
2. **24-layer model** (full): [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/zwu797_connect_hkust-gz_edu_cn/IQAt-hupnydGSYmC6Grp-bwcAVhEN9IOBaSMcqdf4LuWH-w?e=WmsKwV)

After downloading, organize your checkpoints directory as follows:

```bash
metric_checkpoints/
├── pvsm_finetuned_full.pt          # Our trained full 24-layer model
├── pvsm_finetuned_small.pt         # Our trained smaller 12-layer model
├── dinov3-vitb16-pretrain-lvd1689m # DINOv3 Checkpoint
│   ├── config.json
│   ├── LICENSE.md
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── README.md
├── imagenet-vgg-verydeep-19.mat    # (Optional) for training
└── map-anything                    # (Optional) for dataset generation
    ├── config.json
    ├── model.safetensors
    └── README.md
```

### Interactive Demo
For a quick interactive demo, please follow the [instruction](https://github.com/wuzirui/pvsm/blob/master/example/README.md) and unzip the downloaded [example data](https://hkustgz-my.sharepoint.com/:u:/g/personal/zwu797_connect_hkust-gz_edu_cn/IQB3_CSQ7YUbRJ0uowsxovCcAWs5gCP7qmVYkTKsnxnzQGc?e=iTozmL) (22.3 MB) to your local machine.

To launch the interactive web-based demo:

```bash
torchrun --nproc_per_node 1 --standalone viser_demo.py --config-name runs/pvsm_finetuned_small
```

The demo will start a web server. Open your browser and navigate to the displayed URL to interact with the model.

**System Requirements:**
- Small model: ~2.5GB VRAM
- Full model: ~3.0GB VRAM

Note: The rendering quality in [viser](https://github.com/nerfstudio-project/viser) is compressed.

#### Running Inference

To run inference on a dataset:

```bash
python inference.py --config-name runs/pvsm_finetuned_small
```

Or for the full model:

```bash
python inference.py --config-name runs/pvsm_finetuned_full
```


### Training

To train the model:

```bash
torchrun --nproc_per_node <num_gpus> train.py --config-name runs/pvsm_finetuned_small
```

**Configuration:**
- Training configurations are located in `configs/runs/`
- Model configurations are in `configs/model/`
- Dataset configurations are in `configs/dataset/`

**API Keys:**
Before training, create `configs/api_keys.yaml` with your WandB API key:

```yaml
wandb: YOUR_WANDB_KEY
```

You can use `configs/api_keys_example.yaml` as a template.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{wu_pvsm_2026,
  title={From Rays to Projections: Better Inputs for Feed-Forward View Synthesis},
  author={Wu, Zirui and Jiang, Zeren and Oswald, Martin R. and Song, Jie},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgement

This work is built upon [LVSM](https://github.com/Haian-Jin/LVSM)'s code base.
