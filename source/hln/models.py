"""Model definitions.

HybridLungNet is copied verbatim from the original training notebook so that the
existing checkpoint (models/best_hybrid_fold2.pth) loads unchanged.

NoduleDetector3D is new: a small 3-D U-Net style CenterNet detector. It takes a
1 mm isotropic CT crop and outputs, at stride 2:
   heat  : nodule-centre heat-map (sigmoid)
   off   : sub-voxel offset of the centre (3 channels)
   size  : log nodule diameter in mm (1 channel)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ HybridLungNet (original)
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SEBlock3D(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm3d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x), inplace=True)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
                                 nn.Linear(channels // reduction, channels), nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.mlp(self.avg_pool(x).view(b, c)) + self.mlp(self.max_pool(x).view(b, c))
        return x * w.view(b, c, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DualAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        return self.sa(self.ca(x))


class HybridLungNet(nn.Module):
    def __init__(self, aux_features=5, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.layer1 = ResBlock3D(32, 64, stride=2)
        self.layer2 = ResBlock3D(64, 128, stride=2)
        self.layer3 = ResBlock3D(128, 256, stride=2)
        self.layer4 = ResBlock3D(256, 256, stride=2)
        self.attn1 = DualAttention3D(64)
        self.attn2 = DualAttention3D(128)
        self.attn3 = DualAttention3D(256)
        self.attn4 = DualAttention3D(256)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.aux_net = nn.Sequential(nn.Linear(aux_features, 64), nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 32))
        self.classifier = nn.Sequential(nn.Linear(256 + 32, 256), nn.ReLU(inplace=True), self.dropout,
                                        nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x, aux):
        x = self.stem(x)
        x = self.attn1(self.layer1(x))
        x = self.attn2(self.layer2(x))
        x = self.attn3(self.layer3(x))
        x = self.attn4(self.layer4(x))
        x = self.pool(x).flatten(1)
        return self.classifier(torch.cat([x, self.aux_net(aux)], dim=1))


def mc_dropout_predict(model, x, aux, n=20):
    """MC-dropout: only Dropout layers are put in train mode. (The original code
    called model.train(), which ALSO switches BatchNorm to batch statistics; with a
    batch of one patch this changes the prediction itself, not just its spread.)"""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(model(x, aux)).flatten() for _ in range(n)])
    model.eval()
    return p.mean(0), p.std(0)


# ============================================================ Detector (new)
def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv3d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))


class DRes(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1 = conv_bn(i, o, s)
        self.c2 = nn.Sequential(nn.Conv3d(o, o, 3, padding=1, bias=False), nn.BatchNorm3d(o))
        self.sc = nn.Identity() if (i == o and s == 1) else nn.Sequential(nn.Conv3d(i, o, 1, stride=s, bias=False), nn.BatchNorm3d(o))

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.sc(x), inplace=True)


class NoduleDetector3D(nn.Module):
    def __init__(self, w=(24, 48, 96, 160)):
        super().__init__()
        self.stem = conv_bn(1, 16)
        self.e1 = nn.Sequential(DRes(16, w[0], 2), DRes(w[0], w[0]))        # /2
        self.e2 = nn.Sequential(DRes(w[0], w[1], 2), DRes(w[1], w[1]))      # /4
        self.e3 = nn.Sequential(DRes(w[1], w[2], 2), DRes(w[2], w[2]))      # /8
        self.e4 = nn.Sequential(DRes(w[2], w[3], 2), DRes(w[3], w[3]))      # /16
        self.u3 = DRes(w[3] + w[2], w[2])
        self.u2 = DRes(w[2] + w[1], w[1])
        self.u1 = DRes(w[1] + w[0], w[0])
        self.head = nn.Sequential(conv_bn(w[0], 32), nn.Conv3d(32, 5, 1))
        nn.init.constant_(self.head[-1].bias[:1], -4.6)  # prior p~0.01 for heat-map

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        x1 = self.e1(self.stem(x))
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        d3 = self.u3(torch.cat([self.up(x4, x3), x3], 1))
        d2 = self.u2(torch.cat([self.up(d3, x2), x2], 1))
        d1 = self.u1(torch.cat([self.up(d2, x1), x1], 1))
        out = self.head(d1)
        return out[:, :1], out[:, 1:4], out[:, 4:5]   # heat logits, offset, log-size
