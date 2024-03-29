import torch
import torch.nn as nn
import torch.nn.functional as F

class TSC_MLP(nn.Module):
    def __init__(self, input_size: int = 0, output_size: int = 0):
        if input_size == 0 or output_size == 0:
            raise ValueError("Requiring input_size and output_size.")
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 8*output_size),
            nn.ReLU(),
            nn.Linear(8*output_size, output_size),
            )

    def forward(self, input):
        return self.model(input)