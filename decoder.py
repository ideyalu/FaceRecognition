import torch

LATENT_SPACE_SIZE = 128

class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, bn=False):
        super(DecoderBlock, self).__init__()
        if bn:
            self.head = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, hidden_channels[0], 2, 2, 0),
                torch.nn.BatchNorm2d(num_features=in_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(dropout),
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, hidden_channels[0], 2, 2, 0),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(dropout),
            )

        hidden_layers = []
        last_channel = hidden_channels[0]
        for hidden_channel in hidden_channels:
            hidden_layers.append(torch.nn.Conv2d(last_channel, hidden_channel, kernel_size=(3, 3), stride=1, padding=1))
            hidden_layers.append(torch.nn.BatchNorm2d(num_features=hidden_channel))
            hidden_layers.append(torch.nn.ReLU())
            last_channel = hidden_channel

        self.hidden = torch.nn.Sequential(*hidden_layers)

        self.tail = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(last_channel, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(),
        )
    
    def forward(self, input):
        head = self.head(input)
        hidden = self.hidden(head)
        tail = self.tail(head + hidden)
        return tail


class DownsampleBlock(torch.nn.Module):

    def __init__(self, in_channels, channels, dropout=0.3):
        super(DownsampleBlock, self).__init__()
        layers = []
        last_channel = in_channels
        for channel in channels:
            layers.append(torch.nn.Conv2d(last_channel, channel, kernel_size=(2, 2), stride=2, padding=0))
            layers.append(torch.nn.ReLU())
            last_channel = channel

        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, input):
        output = self.model(input)
        return output


class UpsampleBlock(torch.nn.Module):

    def __init__(self, in_channels, channels, dropout=0.3):
        super(UpsampleBlock, self).__init__()
        layers = []
        last_channel = in_channels
        for channel in channels:
            layers.append(torch.nn.ConvTranspose2d(last_channel, channel, 2, 2, 0, bias=False))
            layers.append(torch.nn.ReLU())
            last_channel = channel

        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, input):
        output = self.model(input)
        return output


class Decoder(torch.nn.Module):

    def __init__(self, latent_space_size):
        super(Decoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(latent_space_size, 512, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.ReLU(),

            DecoderBlock(512, [768, 768, 768], 512), 

            UpsampleBlock(512, [256, 128]),

            DecoderBlock(128, [196, 196, 196], 128), 

            UpsampleBlock(128, [64, 32]),

            DecoderBlock(32, [48, 48, 48], 32), 
            
            UpsampleBlock(32, [16, 8]),

            DecoderBlock(8, [12, 12, 12], 8), 

            torch.nn.Conv2d(8, 3, kernel_size=(1, 1), stride=1, padding=0),   

            torch.nn.Tanh(),
        )

    def forward(self, input):
        return self.model(input)
