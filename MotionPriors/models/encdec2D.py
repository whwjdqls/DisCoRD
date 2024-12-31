import torch
import torch.nn as nn
import torch.nn.functional as F

from .Resnet import ResConv2DBlock


class Downsample2D(nn.Module):
    def __init__(self, in_channels, conv_kernel_size, with_conv=True, down_dim="both"):
        """
        Downsampling module for 2D data (e.g., spatial and/or temporal dimensions).
        Args:
        - in_channels (int): Number of input channels.
        - conv_kernel_size (int): Size of the convolutional kernel (3 or 5).
        - with_conv (bool): If True, use convolutional downsampling; otherwise, use max pooling.
        - down_dim (str): Specifies which dimension(s) to downsample ("both", "temporal", or "spatial").
        """
        super().__init__()
        self.with_conv = with_conv
        self.down_dim = down_dim
        assert conv_kernel_size in [3, 5], "Kernel size must be 3 or 5."
        assert down_dim in ["both", "temporal", "spatial"], "down_dim must be 'both', 'temporal', or 'spatial'."
        if down_dim == "both":
            self.stride = (2, 2)
        elif down_dim == "temporal":
            self.stride = (1, 2)
        elif down_dim == "spatial":
            self.stride = (2, 1)

        if self.with_conv:
            self.conv_kernel_size = conv_kernel_size
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=conv_kernel_size, stride=self.stride, padding=0
            )
        else:
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        """
        Forward pass of the downsampling module.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, C, S, T).

        Returns:
        - torch.Tensor: Output tensor after downsampling.
        """
        if self.with_conv:
            # Get input dimensions
            N, C, S, T = x.shape
            K_h, K_w = self.conv_kernel_size, self.conv_kernel_size
            S_h, S_w = self.stride
            # Calculate output dimensions
            if self.down_dim == "both":
                out_S = S // 2
                out_T = T // 2
            elif self.down_dim == "spatial":
                out_S = S // 2
                out_T = T
            elif self.down_dim == "temporal":
                out_S = S
                out_T = T // 2
            # Calculate necessary padding
            pad_h = max((out_S - 1) * S_h + K_h - S, 0)
            pad_w = max((out_T - 1) * S_w + K_w - T, 0)

            # Apply padding (left, right, top, bottom)
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            x = F.pad(x, padding)
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


class Upsample2D(nn.Module):
    def __init__(self, in_channels, conv_kernel_size, with_conv=True, up_dim="both"):
        """
        Upsampling module for 2D data (e.g., spatial and/or temporal dimensions).

        Args:
        - in_channels (int): Number of input channels.
        - conv_kernel_size (int): Size of the convolutional kernel (3 or 5).
        - with_conv (bool): If True, apply a convolution after upsampling.
        - up_dim (str): Specifies which dimension(s) to upsample ("both", "temporal", or "spatial").
        """
        super().__init__()
        self.with_conv = with_conv
        self.up_dim = up_dim
        assert conv_kernel_size in [3, 5], "Kernel size must be 3 or 5."
        assert up_dim in ["both", "temporal", "spatial"], "up_dim must be 'both', 'temporal', or 'spatial'."

        # Determine scale factors for upsampling
        if up_dim == "both":
            self.scale_factor = (2, 2)
        elif up_dim == "temporal":
            self.scale_factor = (1, 2)
        elif up_dim == "spatial":
            self.scale_factor = (2, 1)
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=conv_kernel_size, stride=1, padding=0)

            if conv_kernel_size == 3:
                self.pad = nn.ZeroPad2d((1, 1, 1, 1))

            elif conv_kernel_size == 5:
                self.pad = nn.ZeroPad2d((2, 2, 2, 2))

    def forward(self, x):
        """
        Forward pass of the upsampling module.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, C, S, T).

        Returns:
        - torch.Tensor: Output tensor after upsampling.
        """
        # Upsample using nearest neighbor interpolation
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.with_conv:
            x = self.pad(x)
            x = self.conv(x)
        return x


class Encoder2D(nn.Module):
    def __init__(
        self,
        input_dim=12,
        out_dim=32,  # latent dim
        hidden_size=256,
        t_quant_factor=2,
        s_quant_factor=2,
        s_quant_length=5,
        s_seq_len=21,
        down_conv=True,  # wheter to use conv in downsampling
        down_kernel_size=3,  # kernel size for downsampling conv # if down_conv is False, this is ignored
        activation="relu",
        norm=None,
    ):
        super().__init__()
        for _ in range(s_quant_factor):
            s_seq_len //= 2
        assert s_seq_len == s_quant_length, "s_seq_len and s_quant_length does not match"

        self.quantize_both = min(t_quant_factor, s_quant_factor)
        self.quantize_t = t_quant_factor - self.quantize_both
        self.quantize_s = s_quant_factor - self.quantize_both

        blocks = []
        blocks.append(nn.Conv2d(input_dim, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i, quantize_layer in enumerate([self.quantize_t, self.quantize_s]):
            if i == 0:
                down_dim = "temporal"
            elif i == 1:
                down_dim = "spatial"
            for j in range(quantize_layer):
                input_dim = hidden_size
                block = nn.Sequential(
                    Downsample2D(hidden_size, conv_kernel_size=down_kernel_size, with_conv=down_conv, down_dim=down_dim),
                    ResConv2DBlock(
                        in_channels=hidden_size, out_channels=hidden_size, norm=norm, activation=activation
                    ),
                )
                blocks.append(block)

        for i in range(self.quantize_both):
            input_dim = hidden_size
            block = nn.Sequential(
                Downsample2D(hidden_size, conv_kernel_size=down_kernel_size, with_conv=down_conv, down_dim="both"),
                ResConv2DBlock(in_channels=hidden_size, out_channels=hidden_size, norm=norm, activation=activation),
            )
            blocks.append(block)
        blocks.append(nn.Conv2d(hidden_size, out_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder2D(nn.Module):
    def __init__(
        self,
        input_dim=32,
        out_dim=12,
        hidden_size=256,
        t_quant_factor=2,
        s_quant_factor=2,
        s_quant_length=5,
        s_seq_len=21,
        up_conv=True,  # wheter to use conv in up
        up_kernel_size=3,  # kernel size for upsampling conv # if down_conv is False, this is ignored
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.s_seq_len = s_seq_len
        for _ in range(s_quant_factor):
            s_seq_len //= 2
        assert s_seq_len == s_quant_length, "s_seq_len and s_quant_length does not match"

        self.quantize_both = min(t_quant_factor, s_quant_factor)
        self.quantize_t = t_quant_factor - self.quantize_both
        self.quantize_s = s_quant_factor - self.quantize_both

        blocks = []

        blocks.append(nn.Conv2d(input_dim, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(self.quantize_both):  # upsample both
            input_dim = hidden_size
            block = nn.Sequential(
                ResConv2DBlock(in_channels=hidden_size, out_channels=hidden_size, norm=norm, activation=activation),
                Upsample2D(hidden_size, conv_kernel_size=up_kernel_size, with_conv=up_conv, up_dim="both"),
            )
            blocks.append(block)

        for i, quantize_layer in enumerate([self.quantize_t, self.quantize_s]):
            if i == 0:
                up_dim = "temporal"
            elif i == 1:
                up_dim = "spatial"
            for j in range(quantize_layer):
                input_dim = hidden_size
                block = nn.Sequential(
                    ResConv2DBlock(
                        in_channels=hidden_size, out_channels=hidden_size, norm=norm, activation=activation
                    ),
                    Upsample2D(hidden_size, conv_kernel_size=up_kernel_size, with_conv=up_conv, up_dim=up_dim),
                )
                blocks.append(block)

        self.model = nn.Sequential(*blocks)
        self.post_model = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, out_dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.model(x)
        Bs, C, S, T = x.shape
        if S != self.s_seq_len:
            x = F.interpolate(x, size=(self.s_seq_len, T), mode="bilinear")
        x = self.post_model(x)
        return x


if __name__ == "__main__":
    model = Encoder2D()
    decoder = Decoder2D()
    print(model)
    print(decoder)
    x = torch.randn(2, 12, 21, 64)
    print(x.shape)
    out = model(x)
    print(out.shape)
