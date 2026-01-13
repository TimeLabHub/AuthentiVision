# AuthentiVision 🔍

<div align="center">

<img src="assets/img_1.jpg" alt="Logo" width="300"/>


**State-of-the-art Face Authentication Model for Detecting AI-Generated Images**

[Huggingface](https://huggingface.co/haijian06/AuthentiVision) | [Data](https://huggingface.co/datasets/haijian06/RealAI_Faces) | [Demo](https://huggingface.co/spaces/haijian06/TrueFace) | [Tech Blog](https://timelabhub.github.io/)

</div>
</div>

## 🎯 Real vs. AI-Generated Face Comparison

<div align="center">
<table>
<tr>
<td><b>Real Face</b></td>
<td><b>AI-Generated Face</b></td>
</tr>
<tr>
<td>
<img src="assets/real_face.jpg" alt="Real Face" width="200"/>
</td>
<td>
<img src="assets/ai_face.jpg" alt="AI-Generated Face" width="200"/>
</td>
</tr>
<tr>
<td>
<img src="assets/real_face_2.jpg" alt="Real Face" width="200"/>
</td>
<td>
<img src="assets/ai_face_2.jpg" alt="AI-Generated Face" width="200"/>
</td>
</tr>
</table>
</div>

## 🌟 Features

- High accuracy in distinguishing real faces from AI-generated ones
- Multiple feature extraction techniques for robust detection
- Easy-to-use API for quick integration
- Lightweight and efficient inference

## 🚀 Quick Start

```bash
git clone https://github.com/TimeLabHub/AuthentiVision.git
cd AuthentiVision
pip install -r authentivision/requirements.txt
```

> **Note for Server Environments**: If you are running on a server without a display (e.g., Linux terminal), you may need to install the headless version of OpenCV to avoid `libGL` errors:
> ```bash
> pip uninstall opencv-python
> pip install opencv-python-headless==4.8.1.78
> ```

Download the pretrained model:
```bash
wget -O best_model.pth https://1829447704.v.123pan.cn/1829447704/PaperData/model/best_model.pth
```

Download MFID dataset (Optional, only for training)
```bash
wget https://1829447704.v.123pan.cn/1829447704/PaperData/MFID.zip
unzip MFID.zip -d AuthentiVision_Dataset
```

## 🔮 Prediction

We provide a high-accuracy pretrained model (`best_model.pth`) in the project root, which achieves **>99% accuracy** on the MFID test set. Please ensure you have downloaded it as described in the [Quick Start](#-quick-start) section.

To run prediction using this model:

```bash
cd authentivision
# Predict a single image
python predict.py --input-path /path/to/face.jpg --model-path ../best_model.pth

# Predict a folder of images
python predict.py --input-path /path/to/folder/ --model-path ../best_model.pth
```

Run prediction (simple wrapper script method):

1.Edit `run_prediction.py` and change the `target_path` variable.

2.Run the script:
```bash
python run_prediction.py
```

## 🏋️ Training

To train the model from scratch using the MFID dataset:

1. **Prepare the Dataset**: Ensure you have downloaded and extracted the MFID dataset. The structure should be:
   ```
   AuthentiVision_Dataset/
   ├── real_images/
   └── ai_generated/
   ```

2. **Run Training**:
   You can run the training script with default settings:
   ```bash
   cd authentivision
   python train.py
   ```

   Or specify the dataset path and training parameters:
   ```bash
   python train.py --dataset_root /path/to/AuthentiVision_Dataset --epochs 100 --batch_size 32
   ```

   **Parameters:**
   - `--dataset_root`: Root directory of the dataset. Default: `../AuthentiVision_Dataset`
   - `--epochs`: Number of training epochs. Default: 100
   - `--batch_size`: Batch size for training. Default: 32

   > **Note on Pretrained Models**: The script attempts to download ImageNet pretrained weights for `tf_efficientnetv2_b2` from HuggingFace. If network access is restricted, you can manually download the `model.safetensors` file (e.g., from HuggingFace Hub) and place it in the project root directory. The script will automatically detect and load it if online download fails.

3. **Output**:
   - The best model will be saved as `best_model.pth`.
   - TensorBoard logs will be saved in `runs/advanced_face_detection`.
   - Test set details will be saved to `test_dataset.txt`.

## 📚 Documentation

For detailed documentation, please visit our [tech blog](https://timelabhub.github.io/).

## 🎯 Use Cases(Coming soon)

- Identity verification systems
- Social media content moderation
- Digital forensics
- Security applications
## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📦 Third-Party Datasets and Licenses

This project makes use of several publicly available datasets. Below we list the datasets and their corresponding licenses for transparency and compliance.

### 1️⃣ FFHQ (Flickr-Faces-HQ)

* **Used for**: Real face images in MFID dataset
* **Provider**: NVIDIA
* **Official URL**: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)
* **License**: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

**Notes**:
* The dataset is intended for **non-commercial research use only**
* Attribution is required
* Redistribution of modified versions should preserve the license

**License URL**: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

### 2️⃣ CelebA-HQ

* **Provider**: MMLAB, CUHK
* **License**: **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**

**Notes**:
* Non-commercial research use only
* Face images of celebrities

### 3️⃣ HumanFlux1.1

* **Used for**: Cross-dataset generalization evaluation
* **Provider**: krishnakalyan (Hugging Face)
* **Dataset URL**: [https://huggingface.co/datasets/krishnakalyan3/flux-1.1-v2](https://huggingface.co/datasets/krishnakalyan3/flux-1.1-v2)
* **License**: As specified by the dataset provider (research use).

**Important**:
* At the time of writing, HumanFlux1.1 is released for **research purposes**.
* Please refer to the original dataset card on Hugging Face for the most up-to-date license terms.

### 4️⃣ Rapidata Non-Face Dataset (FLUX1.1)

* **Used for**: Non-face generalization experiments
* **Provider**: Rapidata
* **Dataset URL**: [https://huggingface.co/datasets/Rapidata/117k_human_alignment_flux1.0_V_flux1.1Blueberry](https://huggingface.co/datasets/Rapidata/117k_human_alignment_flux1.0_V_flux1.1Blueberry)
* **License**: Released by Rapidata for research purposes. Please refer to the original dataset page for detailed license terms.

### 5️⃣ AI-Generated Images (StyleGAN, ProGAN, EG3D, Stable Diffusion, Midjourney, FLUX, etc.)

* **Used for**: Synthetic images in MFID
* **Generated by**: Public generative models
* **License considerations**:
  * Generated images follow the **terms of the corresponding generation models**
  * Used **strictly for academic research and benchmarking**
  * No commercial redistribution intended

**Statement**:
All AI-generated images are used solely for academic research and benchmarking purposes, in accordance with the usage policies of the respective generative models.

### ⚠️ Academic Use Disclaimer

The models and code provided in this repository are for **academic research and educational purposes only**. 
- The `MFID` dataset contains images from various sources with specific licensing terms (e.g., CC BY-NC 4.0).
- Users are responsible for ensuring their use of the data and models complies with the original licenses of the source datasets.
- Commercial use of the datasets or models trained on them may be restricted by the original providers.

## 🌟 Acknowledgments

- Thanks to all contributors and researchers in the field
- Special thanks to the open-source community

## 📝 Citation

If you use AuthentiVision in your research or project, please cite our technical blog:

```bibtex
@online{authentivision2024,
    title={AuthentiVision: Finding Yourself in the Real World},
    author={Haijian Wang and Zhangbei Ding and Yefan Niu and Xiaoming Zhang},
    year={2024},
    url={https://timelabhub.github.io/},
    note={Medium blog post}
}
