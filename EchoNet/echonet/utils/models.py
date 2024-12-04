import torch
from foma import get_batch_foma

class VideoNet(torch.nn.Module):
    def __init__(self, vid_model, args):
        super().__init__()
        self.args = args
        self.model = vid_model

    def forward(self, x):
        x = self.model.stem(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.model.fc(x)

        return x

    def forward_mixup(self, x, y):
        x = self.model.stem(x)

        x = self.model.layer1(x)
        x, y = get_batch_foma(self.args, x, y, latent=True)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.model.fc(x)

        return x,y