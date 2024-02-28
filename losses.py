



from timm.utils import accuracy
import torch
import torch.nn.functional as F


def get_reward_matrix(p_e_s, p_e_text, logit_scale = 100.):
    
    p_e_s = p_e_s.transpose(0, 1) # -> F B D
    p_e_s = p_e_s / p_e_s.norm(dim=-1, keepdim=True)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    
    frame_logits = p_e_s @ p_e_text.t() # F B D @ D B -> F B B
    pre_frames, later_frames = torch.chunk(frame_logits, 2, dim = 0)
    diff = later_frames - pre_frames

    reward_matrix = torch.sum(diff, dim=0) * logit_scale
    return reward_matrix

def Loss_DecisionNCE(p_e_s, p_e_text, logit_scale = 100.):
    batch_size = p_e_s.shape[0]
    reward_matrix = get_reward_matrix(p_e_s, p_e_text, logit_scale = logit_scale)
    labels = torch.arange(batch_size, device=reward_matrix.device).long()
    ppt_loss = ( F.cross_entropy(reward_matrix, labels) + \
       F.cross_entropy(reward_matrix.t(), labels) ) / 2
    return ppt_loss





def eval_metric(p_e_s, p_e_text, logit_scale =1.):
    batch_size = p_e_s.shape[0]
    reward_matrix = get_reward_matrix(p_e_s, p_e_text, logit_scale = logit_scale)
    labels = torch.arange(batch_size, device=reward_matrix.device).long()

    def cal_rank_mean(tensor):
        assert len(tensor.shape) == 2
        _, indices = torch.sort(tensor, dim=1)
        rank_of_diagonal_elements_torch = [torch.where(indices[i] == i)[0][0].item() for i in range(tensor.shape[0])]

        return torch.mean(torch.tensor(rank_of_diagonal_elements_torch, dtype=torch.float)) / tensor.shape[0] * 100

    loss = ( F.cross_entropy(reward_matrix, labels) + \
       F.cross_entropy(reward_matrix.t(), labels) ) / 2

    return cal_rank_mean(reward_matrix),  accuracy(reward_matrix, labels)[0], \
    cal_rank_mean(reward_matrix.t()),  accuracy(reward_matrix.t(), labels)[0], loss, torch.diag(reward_matrix).mean()
