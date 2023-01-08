# MIDMs: Matching Interleaved Diffusion Models for Exemplar-based Image Translation

<a href="https://arxiv.org/abs/2209.11047"><img src="https://img.shields.io/badge/arXiv-2209.11047-b31b1b.svg"></a>

 <!-- ## [[Project Page]](https://3dgan-inversion.github.io./) -->

### Official PyTorch implementation of the AAAI 2023 paper

#### Junyoung Seo*, Gyuseong Lee*, Seokju Cho, Jiyoung Lee, Seungryong Kim,

  **equal contribution*

![1](https://ku-cvlab.github.io/MIDMs/resources/qual_celeb.png)

For more information, check out the paper on [Arxiv](https://arxiv.org/abs/2209.11047) or [Project page](https://ku-cvlab.github.io/MIDMs/)

## Preparation

---

### Environmental Settings

Clone the Synchronized-BatchNorm-PyTorch repository.
```
cd models/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ..
```

And, Download the weight of VQ-autoencoder(f=4, VQ, Z=8192, d=3) [here](https://github.com/CompVis/latent-diffusion) and move `model.ckpt` and `config.yaml` to `models/vq-f4`.

After that, please install dependencies.

```bash
conda env create -f environment.yml
conda activate midms
```

Also, if you already have an [LDM](https://github.com/CompVis/latent-diffusion) or [Stable Diffusion Models](https://github.com/CompVis/stable-diffusion) environment, you can use it as well.

### Pretrained Models

We provide finetuned model on CelebA-HQ(edge-to-face). Download the weight [here](https://koreaoffice-my.sharepoint.com/:f:/g/personal/se780_korea_ac_kr/Etx16-dA47pHv8V7GqLKFTgBtvRL67Fn5P5323z9sxQnEA?e=dgDNJp).

Put the weights as followings:

    └── weights

        └── celeba
        
            └── midms_celebA_finetuned.pth
        
            └── pretrained
        
                └── config.yaml
            
                └── model.ckpt

### Datasets

For the datasets, we used the train and validation set provided by [CoCosNet](https://arxiv.org/abs/2004.05571), which can be downloaded from [here](https://github.com/microsoft/CoCosNet).

## Inference

---

Prepare the validation dataset as speicified above, and run inference.py, e.g.,

```bash
python inference.py --benchmark celebahqedge --inference_mode target_fixed --pick 11
```

where `pick` is index of condition image (e.g., sketch). If you want to evaluate the model using the validation set, change the value of `inference_mode` from `target_fixed` to `evaluation`.

## Training

---
Before starting fine-tuning for MIDMs, we first pretrain LDM on the desired dataset following [here](https://github.com/CompVis/latent-diffusion/), or alternatively, the pretrained weights can be obtained from the [model zoo](https://github.com/CompVis/latent-diffusion#model-zoo).

Additionally, pretrained VGG model is required. Please download from the `Training` section of [CoCosNet repository](https://github.com/microsoft/CoCosNet), and move it to `models/`. We used 8 NVIDIA RTX 3090s for finetuning, and it took an average of 5-12 hours per dataset.

 Run `train.py` like:

```bash
torchrun --standalone --nproc_per_node=<NUM_GPU> train.py \
    --benchmark celebahqedge \
    --diffusion_config_path "weights/celeba/pretrained/config.yaml" \
    --diffusion_model_path "weights/celeba/pretrained/model.ckpt" \
    --phase e2e_recurrent --dataroot "/downloaded/dataset/folder" \
    --batch-size <BATCH_SIZE> \
    --snapshots "/path/to/save/results" --warmup_iter 10000
```

### TIP
We discovered that the number of warm-up iterations and the number of training epochs are important when fine-tuning. If training for too long, collapse can occur. In addition, by adjusting the scaling factor of perceptual loss and style loss, the trade-off can be reduced. Finally, the training code is not yet well organized. It is currently being organized and if you encounter any errors or difficulties in implementation, please feel free to contact us.

## Acknowledgement

---

This code implementation is heavily borrowed from the official implementation of [LDM](https://github.com/CompVis/latent-diffusion) and [CoCosNet](https://github.com/microsoft/CoCosNet). We are deeply grateful for all of the projects.

## Bibtex

---

```bibtex
@article{seo2022midms,
  title={MIDMs: Matching Interleaved Diffusion Models for Exemplar-based Image Translation},
  author={Seo, Junyoung and Lee, Gyuseong and Cho, Seokju and Lee, Jiyoung and Kim, Seungryong},
  journal={arXiv preprint arXiv:2209.11047},
  year={2022}
}
```
