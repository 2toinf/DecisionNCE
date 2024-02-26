import clip
from clip.model import CLIP
import numpy as np
from numpy.core.numeric import full
from sklearn.semi_supervised import SelfTrainingClassifier
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torchvision
from torchvision import transforms
from timm.models.layers import Mlp

import torchvision.transforms as T

class CLIPBasedEncoder(nn.Module):
    def __init__(self, modelid="/mnt/lustre/zhengjinliang/PPT/RN50.pt", device="cuda"):
        super().__init__()

        self.modelid = modelid
        self.device = device
        
        # Load CLIP model and transform
        model, cliptransforms = clip.load(modelid, device=self.device, jit=False)
        # CLIP precision
        model.float()
        

        self.model = model 
        self.model.train()
        self.transforms = cliptransforms
        # del self.model.logit_scale 
        # self.model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transforms_tensor = nn.Sequential(
                transforms.Resize(self.model.visual.input_resolution, antialias=None),
                transforms.CenterCrop(self.model.visual.input_resolution),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            )

        self.output_dim = self.model.visual.output_dim   

    def forward(self, visual_input, text_input = None, return_logit_scale = False):
        if type(visual_input) != torch.Tensor:
            print("Warning: Input not tensor, may cause issue with normalization")
            visual_input = self.transforms(visual_input).to(self.device)
        else:
            if torch.max(visual_input) > 10.0:
                visual_input = visual_input / 255.0
            visual_input = self.transforms_tensor(visual_input).to(self.device)

        visual_features = self.model.encode_image(visual_input)
        if text_input is None:
            return visual_features
        text_features = self.model.encode_text(text_input)
        if return_logit_scale:
            return visual_features, text_features, self.model.logit_scale.exp()
        else:
            return visual_features, text_features
        

    
@register_model
def r50_liv(pretrained = None, **kwargs):
    model = CLIPBasedEncoder(modelid="/mnt/lustre/zhengjinliang/PPT/RN50.pt")
    return model
