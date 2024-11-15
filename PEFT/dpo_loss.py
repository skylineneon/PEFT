import torch

class DpoLoss(nn.Module):

    def __init__(self, ignore_index=0, beta=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(self,
                frozen_chosen_y,
                frozen_rejected_y,
                training_chosen_y,
                training_rejected_y,
                chosen_t,
                rejected_t):
        _chosen_mask = (chosen_t != self.ignore_index) & (chosen_t != rejected_t)
        _rejected_mask = (rejected_t != self.ignore_index) & (chosen_t != rejected_t)

        _chosen_size = _chosen_mask.sum()
        _rejected_size = _rejected_mask.sum()

        _chosen_t = chosen_t
        _rejected_t = rejected_t

        _frozen_chosen_y = frozen_chosen_y
        _frozen_rejected_y = frozen_rejected_y

        _training_chosen_y = training_chosen_y
        _training_rejected_y = training_rejected_y

        _show_loss = F.cross_entropy(_training_chosen_y[_chosen_mask], _chosen_t[_chosen_mask])
        _show_loss2 = F.cross_entropy(_frozen_chosen_y[_chosen_mask], _chosen_t[_chosen_mask])

        _frozen_chosen_log_probs = torch.log_softmax(_frozen_chosen_y, dim=-1)
        _frozen_rejected_log_probs = torch.log_softmax(_frozen_rejected_y, dim=-1)
        _training_chosen_log_probs = torch.log_softmax(_training_chosen_y, dim=-1)
        _training_rejected_log_probs = torch.log_softmax(_training_rejected_y, dim=-1)

        _frozen_chosen_score = torch.gather(_frozen_chosen_log_probs, dim=-1, index=_chosen_t[..., None])[..., 0]
        _frozen_rejected_score = torch.gather(_frozen_rejected_log_probs, dim=-1, index=_rejected_t[..., None])[..., 0]
        _training_chosen_score = torch.gather(_training_chosen_log_probs, dim=-1, index=_chosen_t[..., None])[..., 0]
        _training_rejected_score = torch.gather(_training_rejected_log_probs, dim=-1, index=_rejected_t[..., None])[
            ..., 0]

        _chosen_reward = (_training_chosen_score * _chosen_mask).sum(-1) / _chosen_size - (
                _frozen_chosen_score * _chosen_mask).sum(-1) / _chosen_size
        _rejected_reward = (_training_rejected_score * _rejected_mask).sum(-1) / _rejected_size - (
                _frozen_rejected_score * _rejected_mask).sum(-1) / _rejected_size
        # print(F.logsigmoid(self.beta * (_chosen_reward - _rejected_reward)))
        # print(_frozen_chosen_score,"*********")
        _dpo_loss = -torch.mean(F.logsigmoid(self.beta * (0.7 * _chosen_reward - 0.3 * _rejected_reward)))

        return _dpo_loss, _show_loss, _chosen_reward.mean(), _show_loss2