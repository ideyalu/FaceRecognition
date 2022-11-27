import torch
import torchvision

LATENT_SPACE_SIZE = 128

class EncoderVGG19(torch.nn.Module):

    def __init__(self, latent_space_size):
        super(EncoderVGG19, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(4096, latent_space_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(self.model(input))
