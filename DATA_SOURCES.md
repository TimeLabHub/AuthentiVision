# MFID Dataset Sources and Licenses

This document provides detailed information about the datasets used in **MFID (Multidimensional Facial Image Dataset)**, including access links, descriptions, and license terms.

MFID consists of two parts:

1. **In-house subset (redistributable)** archived on Zenodo (DOI):
   **10.5281/zenodo.18504540**
2. **Third-party public datasets and repositories**, which are not redistributed in this repository due to license restrictions.  
   Instead, we provide official access links and descriptions below.

---

## 1. In-House Subset (Public Release)

We publicly release an in-house curated subset of MFID via Zenodo.

- **Repository**: Zenodo
- **DOI**: https://doi.org/10.5281/zenodo.18504540
- **Content**: real face images and AI-generated face images curated for AI-generated face detection tasks.
- **License**: CC BY 4.0

---

## 2. Public Real-Face Datasets

The following datasets are used as real face sources in MFID.

| Dataset Name           | Used Folder Name                                         | Description                                             | Official Link                                        | License                       |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- | ----------------------------- |
| FFHQ (Flickr-Faces-HQ) | `ffhq_256_6000`                                          | High-quality human face dataset(The first 6,000 pieces) | https://github.com/NVlabs/ffhq-dataset               | CC BY-NC 4.0                  |
| CelebA / CelebA-HQ     | `celeba_256_7200`                                        | Celebrity face dataset widely used in vision research   | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html    | Research use / non-commercial |
| Pexels                 | `pexels_resize_256_3200` / `pexels_forceresize_256_1300` | Real-world portrait images collected from Pexels        | https://www.pexels.com/                              | Pexels License                |
| Pixabay                | `pixabay_forceresize_256_400`                            | Real-world portrait images collected from Pixabay       | https://pixabay.com/                                 | Pixabay License               |
| FFHQ_from_hf           | `huggingface_256_9100`                                   | High-quality human face dataset(6000-15099)             | https://huggingface.co/datasets/marcosv/ffhq-dataset | CC BY-NC 4.0                  |

**Note:** Some real-face images were collected from public platforms under their original license terms. Users must ensure compliance with the original providers.

---

## 3. Public AI-Generated Face Datasets / Models

The following AI-generated face sources are used in MFID.  
All synthetic images were generated or collected under the corresponding model usage terms, and are intended for academic research and benchmarking only.

| Generator / Model        | Used Folder Name                  | Description / Notes            | Reference Link                                          |
| ------------------------ | --------------------------------- | ------------------------------ | ------------------------------------------------------- |
| StyleGAN1                | `StyleGAN1_256_1500`              | GAN-generated faces            | https://github.com/NVlabs/stylegan                      |
| StyleGAN2                | `StyleGAN2_256_1500`              | GAN-generated faces            | https://github.com/NVlabs/stylegan2                     |
| StyleGAN3                | `StyleGAN3_256_2000`              | GAN-generated faces            | https://github.com/NVlabs/stylegan3                     |
| ProGAN                   | `ProGAN_256_2000`                 | GAN-generated faces            | https://github.com/tkarras/progressive_growing_of_gans  |
| EG3D                     | `EG3D_256_2000`                   | 3D-aware GAN face generation   | https://github.com/NVlabs/eg3d                          |
| Midjourney v6 (face)     | `midjourneyv6_face_256_1000`      | AI-generated face dataset      | https://huggingface.co/datasets/CortexLM/midjourney-v6  |
| Midjourney v6 (bodyface) | `midjourneyv6_bodyface_256_13000` | AI-generated face/body dataset | https://huggingface.co/datasets/CortexLM/midjourney-v6  |
| DALL·E 2                 | `DALLE2_256_1500`                 | AI-generated face dataset      | https://huggingface.co/datasets/SDbiaseval/dalle2-faces |

---

## 4. License and Usage Disclaimer

- The **Zenodo-released in-house subset** is distributed under **CC BY 4.0**.
- The remaining third-party datasets listed above are subject to their original licenses.
- This project is intended for **academic research and educational use only**.
- Users are responsible for ensuring that any usage complies with the original dataset/model license terms.

---

## 5. Citation

If you use MFID in your research, please cite our Zenodo dataset DOI:

**DOI**: https://doi.org/10.5281/zenodo.18504540

## 6. Rapidata Generalization Benchmark (Cross-domain)

The **Rapidata Generalization Benchmark** is used to evaluate cross-domain robustness and generalization performance.
It contains **4,000 balanced samples**, including **2,000 real images** and **2,000 AI-generated images**, curated from publicly available sources.
This benchmark is constructed as an evaluation subset and is **not redistributed** in this repository due to licensing restrictions.
Instead, we provide the official access links and licensing references below.

### AI-generated subset (2,000)

| Source Name                  | Description                                            | Official Link                                                | License               |
| ---------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ | --------------------- |
| Rapidata Flux 1.1 Preference | AI-generated images from Flux 1.1 preference dataset   | https://huggingface.co/datasets/Rapidata/flux1.1-likert-scale-preference | Refer to dataset card |
| LAION DALL·E 3 Local         | AI-generated images from DALL·E 3 related LAION subset | https://huggingface.co/datasets/ShoukanLabs/LAION-DallE-3-Local | Refer to dataset card |

### Real-image subset (2,000)

| Source Name                         | Description                                 | Official Link                                                | License                      |
| ----------------------------------- | ------------------------------------------- | ------------------------------------------------------------ | ---------------------------- |
| Photo Geometric                     | Real photos with geometric composition      | https://huggingface.co/datasets/Nbardy/photo_geometric       | Refer to dataset card        |
| Kaggle Image Classification Dataset | Public real-world image dataset from Kaggle | https://www.kaggle.com/datasets/duttadebadri/image-classification | Refer to Kaggle dataset page |

**Note:** This benchmark is used for research evaluation only. All images remain subject to their original license terms. Users must follow the licensing requirements and terms of use specified by the original dataset providers.