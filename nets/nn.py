import torch
from mmdet.models.builder import MODELS
from torch.nn.functional import cross_entropy, one_hot, softmax


def build_detector(cfg, train_cfg=None, test_cfg=None):
    args = dict(train_cfg=train_cfg, test_cfg=test_cfg)
    return MODELS.build(cfg, default_args=args)


@MODELS.register_module()
class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    https://arxiv.org/abs/2204.12511
    """

    def __init__(self, epsilon=-1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets, *args, **kwargs):
        ce = cross_entropy(outputs, targets, reduction='none')
        pt = one_hot(targets, outputs.size()[1]) * softmax(outputs, dim=1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()
