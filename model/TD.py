import math

import torch
import torch.nn.functional as F
from torch import nn


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1.0):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        init_ratio = max(init_ratio, 1e-3)
        self.gamma = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor([9.2 * init_ratio], dtype=torch.float32))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out, dtype=torch.float32) * math.pi / channel_out,
            requires_grad=False,
        )
        self.f = nn.Parameter(torch.tensor([0.057 / init_ratio], dtype=torch.float32))
        self.psi = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=False)

    def gen_gabor_bank(self):
        xmax = self.kernel_size // 2
        ymax = self.kernel_size // 2
        y_0 = torch.arange(-ymax, ymax + 1, dtype=torch.float32, device=self.sigma.device)
        x_0 = torch.arange(-xmax, xmax + 1, dtype=torch.float32, device=self.sigma.device)

        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, self.kernel_size, 1)
        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, self.kernel_size)

        theta = self.theta.view(-1, 1, 1, 1)
        sigma = self.sigma.view(-1, 1, 1, 1)
        gamma = self.gamma.view(-1, 1, 1, 1)
        freq = self.f.view(-1, 1, 1, 1)
        psi = self.psi.view(-1, 1, 1, 1)

        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gb = -torch.exp(-0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma ** 2))
        gb = gb * torch.cos(2 * math.pi * freq * x_theta + psi)
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.gen_gabor_bank()
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class AdaptiveScaleGaborPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.gabor_s = nn.Sequential(GaborConv2d(3, 24, 3, 1, 1), nn.Softmax(dim=1))
        self.gabor_m = nn.Sequential(GaborConv2d(3, 24, 5, 1, 2), nn.Softmax(dim=1))
        self.gabor_l = nn.Sequential(GaborConv2d(3, 24, 7, 1, 3), nn.Softmax(dim=1))
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(72, 24, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 3, kernel_size=1, bias=True),
        )

    def forward(self, x):
        feat_s = self.gabor_s(x)
        feat_m = self.gabor_m(x)
        feat_l = self.gabor_l(x)
        concat = torch.cat([feat_s, feat_m, feat_l], dim=1)
        weights = torch.softmax(self.weight_gen(concat), dim=1)
        return torch.cat(
            [feat_s * weights[:, 0:1], feat_m * weights[:, 1:2], feat_l * weights[:, 2:3]],
            dim=1,
        )


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class ResBackbonePyramid(nn.Module):
    def __init__(self, in_channels=72, layers=(2, 2, 10, 2)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        stage2 = self.layer2(x)
        stage3 = self.layer3(stage2)
        final = self.layer4(stage3)
        final = self.bn2(final)
        return {"stage2": stage2, "stage3": stage3, "final": final}

    def forward(self, x):
        return self.forward_features(x)["final"]


class DirEncoder(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.float()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class GlobalSSM(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=3, expand=1):
        super().__init__()
        inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, kernel_size=d_conv, padding=d_conv - 1, groups=inner, bias=True)
        self.A = nn.Parameter(torch.randn(inner, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, inner) * 0.01)
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, : x.shape[1]]
        x_conv = F.silu(x_conv).transpose(1, 2)
        a_soft = F.softmax(self.A, dim=-1)
        h = torch.einsum("bli,ik->blk", x_conv, a_soft)
        y = torch.einsum("blk,ki->bli", h, self.C)
        y = y + self.D.view(1, 1, -1) * x_conv
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + y


class GlobalMambaBranch(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.ssm1 = GlobalSSM(in_channels, d_state=8, d_conv=3, expand=1)
        self.ssm2 = GlobalSSM(in_channels, d_state=8, d_conv=3, expand=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.pre(x)
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.ssm1(seq)
        seq = self.ssm2(seq)
        x = seq.transpose(1, 2).reshape(b, c, h, w)
        x = self.out_proj(x)
        return self.pool(x)

class GaborMambaContextBranch(nn.Module):
    def __init__(self, in_channels=72, out_channels=128):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.mamba = GlobalMambaBranch(out_channels, out_channels)

    def forward(self, x):
        x = self.pre(x)
        return self.mamba(x)

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, h * w)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = x.view(b, c, -1)
        key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        value = x.view(b, c, -1)
        out = torch.bmm(attention, value).view(b, c, h, w)
        return self.gamma * out + x


class OriginalFeatureFusion(nn.Module):
    def __init__(self, input_channels=640):
        super().__init__()
        self.spatial_att = PAM_Module(input_channels)
        self.channel_att = CAM_Module(input_channels)

    def forward(self, x):
        return self.spatial_att(x) + self.channel_att(x)


class ResidualGlobalInjectionFusion(nn.Module):
    def __init__(self, tex_channels=512, dir_channels=128, global_channels=128, out_channels=640):
        super().__init__()
        self.base_reduce = nn.Sequential(
            nn.Conv2d(tex_channels + dir_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.global_embed = nn.Sequential(
            nn.Conv2d(global_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.inject_gate = nn.Sequential(
            nn.Conv2d(tex_channels + dir_channels + global_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.feature_fusion = OriginalFeatureFusion(out_channels)

    def forward(self, tex_feat, dir_feat, global_feat):
        base = self.base_reduce(torch.cat([tex_feat, dir_feat], dim=1))
        global_map = self.global_embed(global_feat)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(tex_feat, 1),
                F.adaptive_avg_pool2d(dir_feat, 1),
                F.adaptive_avg_pool2d(global_feat, 1),
            ],
            dim=1,
        )
        gate = self.inject_gate(pooled)
        fused = base + gate * global_map
        return self.feature_fusion(fused)


class EmbeddingHead(nn.Module):
    def __init__(self, in_channels=640, num_features=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(in_channels * 8 * 8, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


class TexDir(nn.Module):
    def __init__(self, num_features=512):
        super().__init__()
        self.gabor = AdaptiveScaleGaborPreprocess()
        self.tex_backbone = ResBackbonePyramid()
        self.dir_encoder = DirEncoder()
        self.global_branch = GaborMambaContextBranch(72, 128)
        self.fusion = ResidualGlobalInjectionFusion(512, 128, 128, 640)
        self.head = EmbeddingHead(num_features=num_features)

    def forward(self, x, y):
        gabor_feat = self.gabor(x)
        tex_feat = self.tex_backbone(gabor_feat)
        dir_feat = self.dir_encoder(y)
        global_feat = self.global_branch(gabor_feat)
        fused = self.fusion(tex_feat, dir_feat, global_feat)
        return self.head(fused)
