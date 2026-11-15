"""Floor-plan segmentation network of CubiCasa5K ("hg_furukawa_original").

Condensed copy of floortrans/models/hg_furukawa_original.py from
https://github.com/CubiCasa/CubiCasa5k (Kalervo et al., 2019), licensed
CC BY-NC 4.0. Attribute names are unchanged so the official checkpoint
model_best_val_loss_var.pkl loads directly. Removed: random/pre-training
weight init (the checkpoint overwrites all weights) and the heatmap sigmoid
(we only use the room and icon channels).

Output: 44 channels = 21 junction heatmaps | 12 room classes | 11 icon classes.
"""
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        mid = n_out // 2
        self.bn, self.relu = nn.BatchNorm2d(n_in), nn.ReLU(inplace=True)
        self.conv1, self.bn1 = nn.Conv2d(n_in, mid, 1), nn.BatchNorm2d(mid)
        self.conv2, self.bn2 = nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, n_out, 1)
        if n_in != n_out:
            self.conv4 = nn.Conv2d(n_in, n_out, 1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn(x)))
        out = self.conv2(self.relu(self.bn1(out)))
        out = self.conv3(self.relu(self.bn2(out)))
        return out + (self.conv4(x) if hasattr(self, "conv4") else x)


class FloorplanNet(nn.Module):
    """Hourglass encoder-decoder with 4 levels; branch 'a' goes down, branch 'b' is the skip."""

    def __init__(self, n_classes: int = 44):
        super().__init__()
        self.conv1_, self.bn1, self.relu1 = nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(True)
        self.r01, self.maxpool = Residual(64, 128), nn.MaxPool2d(2, 2)
        self.r02, self.r03, self.r04 = Residual(128, 128), Residual(128, 128), Residual(128, 256)

        for l in (1, 2, 3, 4):
            setattr(self, f"maxpool{l}", nn.MaxPool2d(2, 2))
            a = [(256, 256)] * 3 + ([(256, 512), (512, 512)] if l == 4 else [])
            for i, (n_in, n_out) in enumerate(a, 1):
                setattr(self, f"r{l}{i}_a", Residual(n_in, n_out))
            for i, (n_in, n_out) in enumerate([(256, 256), (256, 256), (256, 512)], 1):
                setattr(self, f"r{l}{i}_b", Residual(n_in, n_out))
            setattr(self, f"upsample{l}", nn.ConvTranspose2d(512, 512, 2, stride=2))
            if l > 1:
                setattr(self, f"r{l}_", Residual(512, 512))

        self.conv2_, self.bn2, self.relu2 = nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(True)
        self.conv3_, self.bn3, self.relu3 = nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True)
        self.conv4_ = nn.Conv2d(256, n_classes, 1)
        self.upsample = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=4)

    def _chain(self, x, name, n):
        """Run residual blocks name.format(i=1..n) in sequence, e.g. 'r2{i}_a'."""
        for i in range(1, n + 1):
            x = getattr(self, name.format(i=i))(x)
        return x

    @staticmethod
    def _add(x, y):
        if x.shape != y.shape:
            x = F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False)
        return x + y

    def forward(self, x):
        x = self.maxpool(self.relu1(self.bn1(self.conv1_(x))))
        a = self.r04(self.r03(self.r02(self.r01(x))))

        skips = []
        for l in (1, 2, 3, 4):
            prev = a
            a = self._chain(getattr(self, f"maxpool{l}")(prev), f"r{l}{{i}}_a", 5 if l == 4 else 3)
            skips.append(self._chain(prev, f"r{l}{{i}}_b", 3))

        x = a
        for l in (4, 3, 2):
            x = getattr(self, f"r{l}_")(self._add(getattr(self, f"upsample{l}")(x), skips[l - 1]))
        x = self._add(self.upsample1(x), skips[0])

        x = self.relu2(self.bn2(self.conv2_(x)))
        x = self.relu3(self.bn3(self.conv3_(x)))
        return self.upsample(self.conv4_(x))
