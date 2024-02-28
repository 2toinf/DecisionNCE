import clip
import torch
import torch.nn as nn
import torchvision.transforms as T

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
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
}

def load():
    pass