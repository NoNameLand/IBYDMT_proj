import os

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms.functional import resize


class CountingNet(nn.Module):
    def __init__(self, num_digits, device=torch.device("cpu")):
        super().__init__()

        self.resnet = resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_digits)

        self.to(device)

    @classmethod
    def from_pretrained(cls, num_digits, workdir, device=torch.device("cpu")):
        checkpoint_path = os.path.join(workdir, "weights", "counting", "net.pt")

        model = cls(num_digits, device=device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, image):
        image = resize(image, (224, 224), antialias=True)
        return self.resnet(image)

    def loss_fn(self, image, target):
        output = self.forward(image)

        mse_loss = nn.functional.mse_loss(output, target, reduction="sum")
        accuracy = torch.mean((output.round() == target.round()).float(), dim=-1).sum()
        return mse_loss, accuracy
