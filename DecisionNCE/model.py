import clip
import gdown
import torch
import os
import torch.nn as nn
import torchvision.transforms as T
from typing import Any, Union, List

class CLIPBasedEncoder(nn.Module):
    def __init__(self, modelid="RN50", device="cuda"):
        
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
        self.transforms_tensor = nn.Sequential(
                T.Resize(self.model.visual.input_resolution, antialias=None),
                T.CenterCrop(self.model.visual.input_resolution),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            )

    def encode_image(self, visual_input):
        if type(visual_input) != torch.Tensor:
            visual_input = self.transforms(visual_input).to(self.device)
            if len(visual_input.shape) == 3: visual_input = visual_input.unsqueeze(0)
        else:
            if torch.max(visual_input) > 10.0:
                visual_input = visual_input / 255.0
            visual_input = self.transforms_tensor(visual_input).to(self.device)

        return self.model.encode_image(visual_input)
        
    def encode_text(self, text_input):
        if type(text_input) == str:
            text_input = [text_input]
        if type(text_input) != torch.Tensor:
            text_input = clip.tokenize(text_input).to(self.device)
        return self.model.encode_text(text_input)
    
    def forward(self, visual_input, text_input):
        return self.encode_image(visual_input), self.encode_text(text_input)



_MODELS = {
    "DecisionNCE-T": 
        {
            "modelid": "RN50",
            "download_link": "https://drive.google.com/uc?export=download&id=1W91rPI8z6ot5FmMUE4RyW1hUqaO35Ar5", # TODO: update this link
        },
        
    "DecisionNCE-P": 
        {
            "modelid": "RN50",
            "download_link": "https://drive.google.com/uc?export=download&id=1W91rPI8z6ot5FmMUE4RyW1hUqaO35Ar5", # TODO: update this link
        }
}

# https://drive.google.com/file/d/1W91rPI8z6ot5FmMUE4RyW1hUqaO35Ar5/view?usp=sharing


def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target



def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    print("===========Currently, DecisionNCE only supports the RN50-CLIP model.\
    You are welcome to expand DecisionNCE to more and larger models.==============")
    if name in _MODELS:
        model_path = _download(_MODELS[name]['download_link'], name, os.path.expanduser(f"~/.cache/DecisionNCE"))
    else:
        raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")
    model = CLIPBasedEncoder(_MODELS[name]['modelid'], device)
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    return model
    
    
