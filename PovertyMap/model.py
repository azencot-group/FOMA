import torch.nn as nn
from resnet_multispectral import ResNet18


class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = 1

        # resnet18_ms
        self.enc = ResNet18(args=args, num_classes=1, num_channels=8)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        return self.enc.forward(x)

    def forward_foma(self, x, y):
        return self.enc.forward_foma(x, y)

