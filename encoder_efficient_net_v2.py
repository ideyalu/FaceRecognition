import torch
import torchvision

LATENT_SPACE_SIZE = 128

class EncoderEfficientNetV2(torch.nn.Module):

    def __init__(self, latent_space_size):
        super(EncoderEfficientNetV2, self).__init__()
        self.model = torchvision.models.efficientnet_v2_l(pretrained=True)
        # self.model.classifier[1] = torch.nn.Linear(1280, latent_space_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        return self.tanh(self.model(input))
