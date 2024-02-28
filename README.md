# DecisionNCE: Embodied Multimodal Representations via Implicit Preference LearningnNCE

[[Project Page](https://2toinf.github.io/DecisionNCE/)]  [[Paper]()]

## Introduction

Multimodal pretraining has emerged as an effective strategy for the trinity of goals of representation learning in autonomous robots: 1) extracting both local and global task progression information; 2) enforcing temporal consistency of visual representation; 3) capturing trajectory-level language grounding. Most existing methods approach these via separate objectives, which often reach sub-optimal solutions. In this paper, we propose a universal unified objective that can simultaneously extract meaningful task progression information from image sequences and seamlessly align them with language instructions. We discover that via implicit preferences, where a visual trajectory inherently aligns better with its corresponding language instruction than mismatched pairs, the popular Bradley-Terry model can transform into representation learning through proper reward reparameterizations. The resulted framework,  ***DecisionNCE*** , mirrors an InfoNCE-style objective but is distinctively tailored for decision-making tasks, providing an embodied representation learning framework that elegantly  **extracts both local and global task progression features** , with temporal consistency enforced through implicit time contrastive learning, while **ensuring trajectory-level instruction grounding** via multimodal joint encoding. Evaluation on both simulated and real robots demonstrates that DecisionNCE effectively facilitates diverse downstream policy learning tasks, offering a versatile solution for unified representation and reward learning.

<p align="center"> 
	<img src="assets/images/intro.jpg"width="100%"> 
</p>

## Contents

- [Quick Start](#Quick Start)
- [Dataset](#Dataset)
- [Model Zoo](#Model Zoo)
- [Result](#Result)

## Quick Start

## Dataset

## Model Zoo

## Result
