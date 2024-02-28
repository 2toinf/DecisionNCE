# DecisionNCE: Embodied Multimodal Representations via Implicit Preference LearningnNCE

[[Project Page](https://2toinf.github.io/DecisionNCE/)]  [[Paper]()]

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
conda create -n decisionnce python=3.10 -y
conda activate decisionnce
pip install --upgrade pip
pip install git+https://github.com/openai/CLIP.git
pip install -e .
```

### Usage

```python
import decisionnce
# Load your DecisionNCE model

```

### API

#### `clip.load(name, device)`

Returns the DecisionNCE model specified by the model name returned by `clip.available_models()`. It will download the model as necessary. The `name` argument can also be a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU. 

---



The model returned by `decisionnce.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the DecisionNCE model.

#### `model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of theDecisionNCE model.

#### `model(image: Tensor, text: Tensor)`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.

## Train

### Pretrain

We pretrain vision and language encoder jointly with DecisionNCE-P/T on [EpicKitchen-100](https://epic-kitchens.github.io/2024) dataset. We provide training code and script in this repo. Please follow the instructions below to start training.

1. Data preparation

Please follow the offical instructions and download the EpicKitchen-100 RGB images [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts?tab=readme-ov-file). We reorganized the official annotations into our [training annotations](https://github.com/2toinf/DecisionNCE/blob/main/assets/EpicKitchen-100_train.csv).

2. start training

We use [Slurm](https://slurm.schedmd.com/documentation.html) for multi-node distributed  finetuning.

```bash
sh ./script/slurm_train.sh
```

Please fill in your image and annotation storage path in the specified location of the [script](https://github.com/2toinf/DecisionNCE/blob/main/script/slurm_train.sh).

## Model Zoo

| Models    | Pretaining Methods | Params<br />(M) | Iters | Pretrain ckpt |
| --------- | ------------------- | --------------- | ----- | ------------- |
| RN50-CLIP | DecisionNCE-P       | 386             | 2W    | [link]()         |
| RN50-CLIP | DecisionNCE-T       | 386             | 2W    | [link]()         |

## Evaluation

### Result

1. simulation

<p align="center"> 
	<img src="assets/web/simulation.png"width="40%"> 
</p>

1. real robot

<p align="center"> 
	<img src="assets/images/intro.jpg"width="100%"> 
</p>

### Visualization

We provide our [jupyter notebook]() to visualize the reward curves.
