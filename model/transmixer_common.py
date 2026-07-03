import torch
import torch.nn as nn
import torch.nn.functional as F

from VMamba.models.vmamba import TransMixer


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def build_official_transmixer_block(hidden_dim: int, ssm_d_state: int, mlp_ratio: float = 4.0):
    return TransMixer(
        hidden_dim=hidden_dim,
        ssm_d_state=ssm_d_state,
        mlp_ratio=mlp_ratio,
        channel_first=True,
        forward_type="v2",
    )


class OfficialTransMixerContextBranch(nn.Module):
    """
    Context branch for the TransMixer + Boundary Attention variant.

    The input Gabor feature map is reduced to a compact resolution and then
    passed through official TransMixer blocks imported from the vendored
    VMamba package in the current project.
    """

    def __init__(
        self,
        in_channels: int = 72,
        out_channels: int = 128,
        stem_depth: int = 4,
        blocks: int = 2,
        ssm_d_state: int = 16,
    ):
        super().__init__()
        layers = []
        current_in = in_channels
        for _ in range(stem_depth):
            layers.append(ConvBNAct(current_in, out_channels, kernel_size=3, stride=2))
            current_in = out_channels
        self.stem = nn.Sequential(*layers)
        self.blocks = nn.Sequential(
            *[
                build_official_transmixer_block(
                    hidden_dim=out_channels,
                    ssm_d_state=ssm_d_state,
                )
                for _ in range(blocks)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.boundary_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, boundary_map: torch.Tensor = None) -> torch.Tensor:
        x = self.stem(x)
        if boundary_map is not None:
            boundary_map = F.interpolate(boundary_map, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x * (1.0 + self.boundary_scale * boundary_map)
        x = self.blocks(x)
        x = self.out_proj(x)
        return self.pool(x)


class GatedFeatureFusion(nn.Module):
    def __init__(
        self,
        tex_channels: int = 512,
        dir_channels: int = 128,
        global_channels: int = 128,
        out_channels: int = 640,
    ):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
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

    def forward(self, tex_feat: torch.Tensor, dir_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        tex_feat = self.adaptive_pool(tex_feat)
        dir_feat = self.adaptive_pool(dir_feat)
        global_feat = self.adaptive_pool(global_feat)
        base = self.base_reduce(torch.cat([tex_feat, dir_feat], dim=1))
        global_map = self.global_embed(global_feat)
        gate = self.inject_gate(torch.cat([tex_feat, dir_feat, global_feat], dim=1))
        return base + gate * global_map
