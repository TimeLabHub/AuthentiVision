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
pip install -r requirements.txt
```

```bash
python predict.py --input-path /path/to/some/face.jpg
or
python predict.py --input-path /path/to/a/folder_with_faces/
or
python predict.py --input-path /path/to/face.jpg --model-path /path/to/another_model.pth
```

Run prediction (simple wrapper script method):

1.Edit run_prediction.py and change the target_path variable.

2.Run the script:
```python
python run_prediction.py
```

## 📚 Documentation

For detailed documentation, please visit our [tech blog](https://timelabhub.github.io/).

## 🎯 Use Cases(Coming soon)

- Identity verification systems
- Social media content moderation
- Digital forensics
- Security applications
## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
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
