# DecisionNCE: Embodied Multimodal Representations via Implicit Preference Learning

[[Project Page](https://2toinf.github.io/DecisionNCE/)]  [[Paper](https://arxiv.org/pdf/2402.18137.pdf)]

🔥 **DecisionNCE has been accepted by ICML2024 and selected as outstanding paper at MFM-EAI workshop@ICML2024**

## Introduction

***DecisionNCE*** , mirrors an InfoNCE-style objective but is distinctively tailored for decision-making tasks, providing an embodied representation learning framework that elegantly  **extracts both local and global task progression features** , with temporal consistency enforced through implicit time contrastive learning, while **ensuring trajectory-level instruction grounding** via multimodal joint encoding. Evaluation on both simulated and real robots demonstrates that DecisionNCE effectively facilitates diverse downstream policy learning tasks, offering a versatile solution for unified representation and reward learning.

<p align="center"> 
	<img src="assets/images/intro.jpg"width="100%"> 
</p>

## Contents

- [Quick Start](#quick-start)
- [Train](#Train)
- [Model Zoo](#model-zoo)
- [Evaluation](#Evaluation)

## Quick Start

### Install

1. Clone this repository and navigate to DecisionNCE folder

```bash
git clone https://github.com/2toinf/DecisionNCE.git
cd DecisionNCE
```

2. Install Package

```bash
conda create -n decisionnce python=3.8 -y
conda activate decisionnce
pip install -e .
```

### Usage

```python

import DecisionNCE
import torch
from PIL import Image
# Load your DecisionNCE model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DecisionNCE.load("DecisionNCE-P", device=device)

image = Image.open("Your Image Path Here")
text = "Your Instruction Here"

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    reward = model.get_reward(image, text) # please note that number of image and text should be the same
```

### API

#### `decisionnce.load(name, device)`

Returns the DecisionNCE model specified by the model name returned by `decisionnce.available_models()`. It will download the model as necessary. The `name` argument should be `DecisionNCE-P` or `DecisionNCE-T`

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU.

---

The model returned by `decisionnce.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the DecisionNCE model.

#### `model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of the DecisionNCE model.

## Train

### Pretrain

We pretrain vision and language encoder jointly with DecisionNCE-P/T on [EpicKitchen-100](https://epic-kitchens.github.io/2024) dataset. We provide training code and script in this repo. Please follow the instructions below to start training.

1. Data preparation

Please follow the offical instructions and download the EpicKitchen-100 RGB images [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts?tab=readme-ov-file). And we provide our  [training annotations](https://github.com/2toinf/DecisionNCE/blob/main/assets/EpicKitchen-100_train.csv) reorganized according to the official version

2. start training

We use [Slurm](https://slurm.schedmd.com/documentation.html) for multi-node distributed  finetuning.

```bash
sh ./script/slurm_train.sh
```

Please fill in your image and annotation path in the specified location of the [script](https://github.com/2toinf/DecisionNCE/blob/main/script/slurm_train.sh).

## Model Zoo

| Models    | Pretaining Methods | Params<br />(M) | Iters | Pretrain ckpt                                                                              |
| --------- | ------------------- | --------------- | ----- | ------------------------------------------------------------------------------------------ |
| RN50-CLIP | DecisionNCE-P       | 386             | 2W    | [link](https://drive.google.com/file/d/1LmDHaKMZCv9QT89dWubZ8dRo6qwpVMYo/view?usp=drive_link) |
| RN50-CLIP | DecisionNCE-T       | 386             | 2W    | [link](https://drive.google.com/file/d/14wn2R5ZDNujSq9Tsaeuy6fJNr6zM5l7E/view?usp=drive_link) |

## Evaluation

### Result

1. simulation

<p align="center"> 
	<img src="assets/web/simulation.png"width="40%"> 
</p>

1. real robot

<p align="center"> 
	<img src="assets/web/realrobot.jpg"width="100%"> 
</p>

### Visualization

We provide our [jupyter notebook]() to visualize the reward curves. Please install jupyter notebook first.

```python
conda install jupyter notebook
```

---

TO BE UPDATE

### Citation

If you find our code and paper can help, please cite our paper as:

```
@inproceedings{lidecisionnce,
  title={DecisionNCE: Embodied Multimodal Representations via Implicit Preference Learning},
  author={Li, Jianxiong and Zheng, Jinliang and Zheng, Yinan and Mao, Liyuan and Hu, Xiao and Cheng, Sijie and Niu, Haoyi and Liu, Jihao and Liu, Yu and Liu, Jingjing and others},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
