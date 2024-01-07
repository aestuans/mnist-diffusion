from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class UNetEncoder(nn.Module):
    def __init__(self, feats):
        super().__init__()

        assert len(feats) == 3

        self.block1 = ResidualBlock(1, feats[0])
        self.block2 = ResidualBlock(feats[0], feats[1], stride=2)
        self.block3 = ResidualBlock(feats[1], feats[2], stride=2)
    
    def forward(self, x):
        x = self.block1(x)
        x_skip = x.clone()

        x = self.block2(x)
        x = self.block3(x)

        return x, x_skip

class UNetDecoder(nn.Module):
    def __init__(self, feats):
        super().__init__()

        assert len(feats) == 3

        self.up_block1 = ResidualBlock(feats[0], feats[1])
        self.up_conv1 = nn.ConvTranspose2d(feats[1], feats[1], kernel_size=2, stride=2)

        self.up_block2 = ResidualBlock(feats[1], feats[2])
        self.up_conv2 = nn.ConvTranspose2d(feats[2], feats[2], kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.up_block1(x)
        x = self.up_conv1(x)

        x = self.up_block2(x)
        x = self.up_conv2(x)

        return x

@dataclass
class ModelConfig:
    encoder_feats: list
    decoder_feats: list
    embedding_feats: int

class UNet(nn.Module):
    def __init__(self, config: ModelConfig, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        # make room for time and context embeddings
        self.config.decoder_feats[0] += 2 * config.embedding_feats

        self.encoder = UNetEncoder(self.config.encoder_feats)

        self.time_mlp = MLP(1, self.config.embedding_feats, self.config.embedding_feats)
        self.context_mlp = MLP(num_classes, self.config.embedding_feats, self.config.embedding_feats)

        self.decoder = UNetDecoder(self.config.decoder_feats)

        self.final_block = ResidualBlock(self.config.encoder_feats[0] + self.config.decoder_feats[-1], 
                                         self.config.decoder_feats[-1])
        self.final_conv = nn.Conv2d(self.config.decoder_feats[-1], 1, kernel_size=1)

    def forward(self, x, t, c, cmask=None):
        """
        x: (batch_size, C, H, W) float tensor: input image
        t: (batch_size,) long tensor: time step
        c: (batch_size,) long tensor: context cue (class label)
        cmask: (batch_size,) data points for which no context cue will be passed to the model
        """
        if cmask is None:
            cmask = torch.zeros_like(c)
        c_embed = F.one_hot(c, num_classes=self.num_classes).float()

        return self._forward(x, t, c_embed, cmask)

    def _forward(self, x, t, c_embed, cmask):
        """
        x: (batch_size, C, H, W) float tensor: input image
        t: (batch_size,) long tensor: time step
        c: (batch_size, num_classes) float tensor: one-hot encoded context cue
        cmask: (batch_size,) data points for which no context cue will be passed to the model
        """
        assert x.dim() == 4
        batch_size = x.size(0)

        x, x_skip = self.encoder(x)

        t_embed = self.time_mlp(t)
        c_embed = self.context_mlp(c_embed)
        c_embed[cmask == 1, :] = 0
        # reshape t_embed and c_embed to the resolution of x and concatenate
        t_embed = t_embed.view(batch_size, -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        c_embed = c_embed.view(batch_size, -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t_embed, c_embed], dim=1)

        x = self.decoder(x)

        # Concatenate with skip connection
        x = torch.cat([x, x_skip], dim=1)

        x = self.final_block(x)
        x = self.final_conv(x)

        return x
