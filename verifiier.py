import torch

class Verifier(torch.nn.Module):

    def __init__(self, in_features, dropout=0.2):
        super(Verifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid(),
            
        )

    def forward(self, input):
        return self.model(input)
