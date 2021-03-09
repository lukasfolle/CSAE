import torch
import torch.nn.functional as F
from torch import nn


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


DIMENSION = 3

if DIMENSION == 2:
    Convd = nn.Conv2d
    ConvTransposed = nn.ConvTranspose2d
    MaxPoold = nn.MaxPool2d
    DropOutd = nn.Dropout2d

elif DIMENSION == 3:
    Convd = nn.Conv3d
    ConvTransposed = nn.ConvTranspose3d
    MaxPoold = nn.MaxPool3d
    DropOutd = nn.Dropout3d


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor, dropout_prob=0.1):
        super().__init__()
        self.encoder_first_layer = nn.Sequential(
            Convd(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DropOutd(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            MaxPoold(2, stride=downsample_factor)
        )

        self.encoder_second_layer = nn.Sequential(
            Convd(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DropOutd(dropout_prob) if dropout_prob > 0 else nn.Identity(),
        )

        self.dense_connection = nn.Sequential(
            Convd(out_channels * 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        z_first = self.encoder_first_layer(x)
        z_second = self.encoder_second_layer(z_first)
        z = F.relu(self.dense_connection(torch.cat((z_first, z_second), dim=1)))
        return z


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, activation=None, upsampler=None):
        super().__init__()
        if activation is None:
            Activation = nn.LeakyReLU
        else:
            Activation = activation
        if upsampler is None:
            Upsample = ConvTransposed(in_channels, out_channels, kernel_size=upsample_factor, stride=upsample_factor)
        elif upsampler == "interpolate":
            Upsample = nn.Sequential(Interpolate(scale_factor=upsample_factor, mode="trilinear"),
                                     Convd(in_channels, out_channels, kernel_size=3, padding=1))
        self.decoder_first_layer = nn.Sequential(
            Upsample,
            Activation()
        )
        self.decoder_second_layer = nn.Sequential(
            Convd(out_channels, out_channels, kernel_size=3, padding=1),
            Activation()
        )

    def forward(self, x):
        x_hat_first = self.decoder_first_layer(x)
        x_hat = self.decoder_second_layer(x_hat_first)
        return x_hat


class CAE(nn.Module):
    def __init__(self, num_features=None, downsample_factor=2, num_classes=None, input_dim=None, drop_rate=0.0,
                 output_channels=None):
        super().__init__()
        if num_features is None:
            num_features = [1, 16, 32]
        self.encoder, self.decoder = self.create_en_and_decoder(num_features, downsample_factor, drop_rate,
                                                                output_channels)
        if num_classes is not None:
            bottleneck_dim = [i_dim // (2 ** (len(num_features) - 1)) for i_dim in input_dim]
            self.classifier = nn.Linear(np.max(num_features) * np.prod(bottleneck_dim), num_classes)

    @staticmethod
    def create_en_and_decoder(num_features, downsample_factor, drop_rate, output_channels):
        encoder_blocks = []
        for i in range(len(num_features) - 1):
            encoder_blocks.append(EncoderBlock(num_features[i], num_features[i + 1], downsample_factor, drop_rate))
        decoder_blocks = []
        num_features.reverse()
        for i in range(len(num_features) - 1):
            decoder_blocks.append(DecoderBlock(num_features[i], num_features[i + 1], downsample_factor,
                                               upsampler="interpolate"))
        # One further decoder block
        out_channels = num_features[-1] if output_channels is None else output_channels
        decoder_blocks.append(DecoderBlock(num_features[-1], out_channels, 1, activation=nn.Identity,
                                           upsampler="interpolate"))
        encoder = nn.Sequential(*encoder_blocks)
        decoder = nn.Sequential(*decoder_blocks)
        return encoder, decoder

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.classifier(z.view(z.shape[0], -1))
        x_hat = self.decoder(z)
        return y_hat, x_hat

