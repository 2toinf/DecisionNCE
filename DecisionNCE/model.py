import clip
import torch
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
            print("Warning: Input not tensor, may cause issue with normalization")
            visual_input = self.transforms(visual_input).to(self.device)
        else:
            if torch.max(visual_input) > 10.0:
                visual_input = visual_input / 255.0
            visual_input = self.transforms_tensor(visual_input).to(self.device)

        return self.model.encode_image(visual_input)
        
    def encode_text(self, text_input):
        if type(text_input) != torch.Tensor:
            text_input = clip.tokenize(text_input).to(self.device)
        return self.model.encode_text(text_input)
    
    def forward(self, visual_input, text_input):
        return self.encode_image(visual_input), self.encode_text(text_input)



_MODELS = {
    "DecisionNCE-T": 
        {
            "modelid": "RN50",
            "download_link": "", # TODO: update this link
        },
        
    "DecisionNCE-P": 
        {
            "modelid": "RN50",
            "download_link": "", # TODO: update this link
        }
}

# https://drive.google.com/file/d/1W91rPI8z6ot5FmMUE4RyW1hUqaO35Ar5/view?usp=drive_link

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    print("Currently, DecisionNCE only supports the RN50-CLIP model. \
        You are welcome to expand DecisionNCE to more and larger models.")
    if name in _MODELS:
        model_path = _download(_MODELS[name]['download_link'], download_root or os.path.expanduser("~/.cache/clip"))
    else:
        raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")
    model = CLIPBasedEncoder(_MODELS[name]['modelid'], device)
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    return model
    
    
