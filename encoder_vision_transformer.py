import torch
import torchvision

LATENT_SPACE_SIZE = 128

class EncoderVisionTransformer(torch.nn.Module):

    def __init__(self, latent_space_size):
        super(EncoderVisionTransformer, self).__init__()
        self.model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1', dropout=0.2, attention_dropout=0.2)
        self.model.heads[0] = torch.nn.Linear(768, latent_space_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        return self.tanh(self.model(input))
