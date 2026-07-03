import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import VMamba.models.vmamba as vmamba
from VMamba.models.utils import get_global_local_index
from model.TD import GaborConv2d
from model.transmixer_common import ConvBNAct

DEFAULT_GABOR_KERNELS = (5, 7, 9)
DEFAULT_BAM_DELTA = 0.2
DEFAULT_BAM_ETA = 0.01
DEFAULT_OUTPUT_CHANNELS = 72

class DRAFGate(nn.Module):
    def __init__(self, global_dim: int, local_dim: int, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.global_norm = nn.LayerNorm(global_dim)
        self.local_proj = nn.Linear(local_dim, global_dim)
        self.local_norm = nn.LayerNorm(global_dim)
        hidden_dim = max(16, global_dim // 4)
        self.route = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, global_dim),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        local_proj = self.local_proj(local_feat)
        u = self.global_norm(global_feat) * self.local_norm(local_proj)
        logits = self.route(u)
        if self.training:
            eps = torch.finfo(logits.dtype).eps
            noise = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
            logistic_noise = torch.log(noise) - torch.log1p(-noise)
            return torch.sigmoid((logits + logistic_noise) / self.tau)
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).to(probs.dtype)

    def project_local(self, local_feat: torch.Tensor) -> torch.Tensor:
        return self.local_proj(local_feat)


class AdaptiveScaleGaborPreprocessFlexible(nn.Module):
    """Multi-scale learnable Gabor preprocessing with a fixed 72-channel output."""

    def __init__(
        self,
        kernels=DEFAULT_GABOR_KERNELS,
        in_channels: int = 3,
        out_channels: int = DEFAULT_OUTPUT_CHANNELS,
    ):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels:
            raise ValueError("At least one Gabor kernel is required.")
        if out_channels % len(kernels) != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by {len(kernels)} kernels.")

        self.kernels = kernels
        self.channels_per_scale = out_channels // len(kernels)
        branches = []
        for kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError(f"Gabor kernel size must be odd, got {kernel}.")
            branches.append(
                nn.Sequential(
                    GaborConv2d(
                        in_channels,
                        self.channels_per_scale,
                        kernel,
                        stride=1,
                        padding=kernel // 2,
                        init_ratio=kernel / 3.0,
                    ),
                    nn.Softmax(dim=1),
                )
            )
        self.branches = nn.ModuleList(branches)
        hidden_channels = max(24, len(kernels) * 8)
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, len(kernels), kernel_size=1, bias=True),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        concat = torch.cat(feats, dim=1)
        weights = torch.softmax(self.weight_gen(concat), dim=1)
        weighted = [feat * weights[:, idx : idx + 1] for idx, feat in enumerate(feats)]
        return torch.cat(weighted, dim=1)


class DRAFSS2D(vmamba.SS2D):
    def __init__(self, *args, draf_tau: float = 1.0, draf_sparse_lambda: float = 0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.draf_sparse_lambda = float(draf_sparse_lambda)
        self.last_sparse_loss = None
        self.last_mask_mean = None
        if getattr(self, "mixer", False):
            mix_dim = self.k_group * self.d_inner
            global_dim = int(mix_dim * self.g_ratio)
            local_dim = mix_dim - global_dim
            self.draf_gate = DRAFGate(global_dim=global_dim, local_dim=local_dim, tau=draf_tau)

    def _reset_draf_stats(self, like: torch.Tensor):
        self.last_sparse_loss = like.new_zeros(())
        self.last_mask_mean = like.new_zeros(())

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        force_fp32=False,
        ssoflex=True,
        no_einsum=False,
        selective_scan_backend=None,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode
        assert isinstance(_scan_mode, int)
        self._reset_draf_stats(x)

        # Keep rare cascade mode identical to the vendored implementation.
        if _scan_mode == -1:
            return super().forward_corev2(
                x=x,
                force_fp32=force_fp32,
                ssoflex=ssoflex,
                no_einsum=no_einsum,
                selective_scan_backend=selective_scan_backend,
                scan_mode=scan_mode,
                scan_force_torch=scan_force_torch,
                **kwargs,
            )

        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B_param, C_param, D_param=None, delta_bias=None, delta_softplus=True):
            return vmamba.selective_scan_fn(
                u,
                delta,
                A,
                B_param,
                C_param,
                D_param,
                delta_bias,
                delta_softplus,
                ssoflex,
                backend=selective_scan_backend,
            )

        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = vmamba.cross_scan_fn(
            x,
            in_channel_first=True,
            out_channel_first=True,
            scans=_scan_mode,
            force_torch=scan_force_torch,
        )
        if no_einsum:
            x_dbl = F.conv1d(
                xs.view(B, -1, L),
                self.x_proj_weight.view(-1, D, 1),
                bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                groups=K,
            )
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(
                    dts.contiguous().view(B, -1, L),
                    self.dt_projs_weight.view(K * D, -1, 1),
                    groups=K,
                )
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)

        if self.mixer:
            g_idx, l_idx = get_global_local_index(dts, delta_bias, self.g_ratio)
            batch_idx = torch.arange(B, device=ys.device)[:, None, None]
            token_idx = torch.arange(L, device=ys.device)[None, :, None]

            g_ys = ys[batch_idx, g_idx, token_idx]
            l_ys = ys[batch_idx, l_idx, token_idx]

            g_att = self.global_module(g_ys)
            l_refined = self.local_module(l_ys, H, W)
            mask = self.draf_gate(g_ys, l_ys)
            local_comp = self.draf_gate.project_local(l_ys)

            ys[batch_idx, g_idx, token_idx] = mask * g_att + (1.0 - mask) * local_comp
            ys[batch_idx, l_idx, token_idx] = l_refined

            mask_mean = mask.mean()
            self.last_mask_mean = mask_mean.detach()
            self.last_sparse_loss = self.draf_sparse_lambda * mask_mean

        ys = ys.view(B, K, -1, H, W)
        y = vmamba.cross_merge_fn(
            ys,
            in_channel_first=True,
            out_channel_first=True,
            scans=_scan_mode,
            force_torch=scan_force_torch,
        )

        if getattr(self, "__DEBUG__", False):
            setattr(
                self,
                "__data__",
                dict(A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds, us=xs, dts=dts, delta_bias=delta_bias, ys=ys, y=y, H=H, W=W),
            )

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = out_norm(y)
        return y.to(x.dtype)


class DRAFTransMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = vmamba.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        draf_tau: float = 1.0,
        draf_sparse_lambda: float = 0.001,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim, channel_first=channel_first)
            self.op = DRAFSS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
                compute_attn_matrix_fn=False,
                draf_tau=draf_tau,
                draf_sparse_lambda=draf_sparse_lambda,
            )

        self.drop_path = vmamba.DropPath(drop_path)

        if self.mlp_branch:
            _MLP = vmamba.Mlp if not gmlp else vmamba.gMlp
            self.norm2 = vmamba.LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        return self._forward(input)

    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        if not self.ssm_branch:
            return torch.zeros((), device=device)
        loss = getattr(self.op, "last_sparse_loss", None)
        if loss is None:
            return torch.zeros((), device=device)
        return loss

    def get_mask_mean(self):
        if not self.ssm_branch:
            return None
        return getattr(self.op, "last_mask_mean", None)


class DRAFOfficialTransMixerContextBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 72,
        out_channels: int = 128,
        stem_depth: int = 4,
        blocks: int = 2,
        ssm_d_state: int = 16,
        draf_tau: float = 1.0,
        draf_sparse_lambda: float = 0.001,
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
                DRAFTransMixer(
                    hidden_dim=out_channels,
                    ssm_d_state=ssm_d_state,
                    mlp_ratio=4.0,
                    channel_first=True,
                    forward_type="v2",
                    draf_tau=draf_tau,
                    draf_sparse_lambda=draf_sparse_lambda,
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

    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        losses_ = [block.get_aux_loss(device) for block in self.blocks]
        if not losses_:
            return torch.zeros((), device=device)
        return torch.stack(losses_).sum()

    def get_mask_mean(self):
        values = []
        for block in self.blocks:
            value = block.get_mask_mean()
            if value is not None:
                values.append(value)
        if not values:
            return None
        return torch.stack([v.detach() for v in values]).mean()
