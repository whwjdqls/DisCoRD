import torch.nn as nn

from .Resnet import Resnet1D


class Encoder1D(nn.Module):
    def __init__(
        self,
        input_dim=3,
        out_dim=512,
        hidden_size=512,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_dim, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = hidden_size
            block = nn.Sequential(
                nn.Conv1d(input_dim, hidden_size, filter_t, stride_t, pad_t),
                Resnet1D(hidden_size, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(hidden_size, out_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder1D(nn.Module):
    def __init__(
        self,
        input_dim=3,
        out_dim=512,
        hidden_size=512,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(input_dim, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            block = nn.Sequential(
                Resnet1D(
                    hidden_size, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(hidden_size, hidden_size, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(hidden_size, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(hidden_size, out_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x
