import torch
import torch.nn.functional as F

def get_reward_matrix_P(p_e_s, p_e_text, logit_scale = 100.):
    p_e_s = p_e_s.transpose(0, 1) # -> F B D
    p_e_s = p_e_s / p_e_s.norm(dim=-1, keepdim=True)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    frame_logits = p_e_s @ p_e_text.t() # F B D @ D B -> F B B
    pre_frames, later_frames = torch.chunk(frame_logits, 2, dim = 0)
    diff = later_frames - pre_frames
    reward_matrix = torch.sum(diff, dim=0) * logit_scale
    return reward_matrix

def get_reward_matrix_T(p_e_s, p_e_text, logit_scale = 100.):
        p_e_s = p_e_s.transpose(0, 1) # -> F B D
        diff = p_e_s[1:] - p_e_s[:-1] # -> F-1, B, D
        diff = diff / (diff.norm(dim=-1, keepdim=True) + 1e-6)
        p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
        frame_logits = diff @ p_e_text.t() * logit_scale # F-1 B D @ D B -> F-1 B B
        reward_matrix = torch.sum(frame_logits, dim=0)
        return reward_matrix

class DecisionNCELoss(torch.nn.Module):
    def __init__(self, logit_scale = 100, loss_type = "DecionNCE-T", **kwargs):
        
        super().__init__()
        self.logit_scale = logit_scale
        self.loss_type = loss_type
        assert self.loss_type in ['DecionNCE-T', 'DecionNCE-P'], f"Unknow loss type: {loss_type}"
        
    def forward(self, visual_features, text_features):
        batch_size = visual_features.shape[0]
        
        if self.loss_type == 'DecionNCE-T':
            reward_matrix = get_reward_matrix_T(visual_features, text_features, logit_scale = self.logit_scale)
        elif self.loss_type == 'DecionNCE-P':
            reward_matrix = get_reward_matrix_P(visual_features, text_features, logit_scale = self.logit_scale)
        else:
            raise NotImplementedError
    
        labels = torch.arange(batch_size, device=reward_matrix.device).long()
        return ( F.cross_entropy(reward_matrix, labels) + \
        F.cross_entropy(reward_matrix.t(), labels) ) / 2

