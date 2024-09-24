import torch

from torch import nn

import torchvision

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ImgNetWrapper(nn.Module):

    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.device = device    
        self.resnet = torchvision.models.resnet18(pretrained=True).to(device)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])



    def forward(self, img, output_stage=""):
        # if image is 3xHxW, then convert to 1x3xHxW
        img = F.to_tensor(img).to(self.device)
        if(img.shape[0] == 3):
            img = img.unsqueeze(0)
        img = self.transform(img)
        out = self.resnet(img)
        return out