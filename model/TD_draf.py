import torch
import torch.nn as nn

from model.TD import DirEncoder, EmbeddingHead, ResBackbonePyramid
from model.boundary_attention import BoundaryAttentionLite
from model.transmixer_common import GatedFeatureFusion
from model.draf_transmixer import (
    DEFAULT_BAM_DELTA,
    DEFAULT_BAM_ETA,
    DEFAULT_GABOR_KERNELS,
    AdaptiveScaleGaborPreprocessFlexible,
    DRAFOfficialTransMixerContextBranch,
)

class DRAFTexDir(nn.Module):
    def __init__(
        self,
        num_features: int = 512,
        draf_tau: float = 1.0,
        draf_sparse_lambda: float = 0.001,
        gabor_kernels=DEFAULT_GABOR_KERNELS,
        bam_delta: float = DEFAULT_BAM_DELTA,
        bam_eta: float = DEFAULT_BAM_ETA,
        **kwargs,
    ):
        super().__init__()
        self.gabor_kernels = tuple(gabor_kernels)
        self.bam_delta = float(bam_delta)
        self.bam_eta = float(bam_eta)
        self.gabor_preprocess = AdaptiveScaleGaborPreprocessFlexible(self.gabor_kernels)
        self.backbone = ResBackbonePyramid(in_channels=72)
        self.dir_encoder = DirEncoder()
        self.context_branch = DRAFOfficialTransMixerContextBranch(
            in_channels=72,
            out_channels=128,
            stem_depth=4,
            blocks=2,
            draf_tau=draf_tau,
            draf_sparse_lambda=draf_sparse_lambda,
        )
        self.gated_fusion = GatedFeatureFusion(tex_channels=512, dir_channels=128, global_channels=128, out_channels=640)
        self.boundary_attention = BoundaryAttentionLite(
            in_channels=640,
            patch_size=3,
            delta=self.bam_delta,
            eta=self.bam_eta,
            downsample=1,
        )
        self.embedding = EmbeddingHead(in_channels=640, num_features=num_features)

    def forward(self, x, y):
        gabor = self.gabor_preprocess(x)
        tex_feat = self.backbone.forward_features(gabor)["final"]
        dir_feat = self.dir_encoder(y)
        global_feat = self.context_branch(gabor)
        fused = self.gated_fusion(tex_feat, dir_feat, global_feat)
        enhanced, _ = self.boundary_attention(fused)
        return self.embedding(enhanced)

    def get_aux_loss(self) -> torch.Tensor:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return self.context_branch.get_aux_loss(device)

    def get_draf_mask_mean(self):
        return self.context_branch.get_mask_mean()

TexDir = DRAFTexDir
