# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper Loss.
    输入为B×C×H×W。
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        dot = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)

        cos = dot / (pred_norm * target_norm + self.eps)
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)

        angle = torch.acos(cos)
        return torch.mean(angle)


class SpectralGradientLoss(nn.Module):
    """
    光谱梯度一致性损失，用于约束相邻波段变化趋势。
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_grad = target[:, 1:, :, :] - target[:, :-1, :, :]
        return self.l1(pred_grad, target_grad)


class VQCommitmentLoss(nn.Module):
    """
    VQ-AE中的codebook损失。
    """
    def __init__(self, beta: float = 0.25):
        super().__init__()
        self.beta = beta

    def forward(self, z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        return codebook_loss + self.beta * commitment_loss


class DataConsistencyLoss(nn.Module):
    """
    数据一致性损失。

    uniform 模式：
        pred_hsi 均匀抽取波段后接近 HR-MSI；

    srf 模式：
        pred_hsi 经过 SRF 投影后接近 HR-MSI。
    """

    def __init__(self, scale_ratio: int, n_select_bands: int, srf_weights=None):
        super().__init__()

        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.l1 = nn.L1Loss()

        if srf_weights is None:
            self.register_buffer("srf_weights", torch.empty(0))
        else:
            self.register_buffer(
                "srf_weights",
                torch.tensor(srf_weights, dtype=torch.float32),
            )

    def forward(
        self,
        pred_hsi: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = pred_hsi.shape

        pred_lr = F.interpolate(
            pred_hsi,
            size=lr_hsi.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        if self.srf_weights.numel() > 0:
            pred_msi = torch.einsum("bchw,mc->bmhw", pred_hsi, self.srf_weights)

        else:
            indices = torch.linspace(
                0,
                c - 1,
                self.n_select_bands,
                device=pred_hsi.device,
            ).round().long()

            pred_msi = pred_hsi.index_select(1, indices)

        loss_lr = self.l1(pred_lr, lr_hsi)
        loss_msi = self.l1(pred_msi, hr_msi)

        return loss_lr + loss_msi

class SpectralDirectionLoss(nn.Module):
    """
    光谱方向损失。
    与SAMLoss目标一致，但直接优化 1-cos，相比acos更平滑。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1, keepdim=True) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1, keepdim=True) + self.eps)

        pred_dir = pred / (pred_norm + self.eps)
        target_dir = target / (target_norm + self.eps)

        cos = torch.sum(pred_dir * target_dir, dim=1)
        loss = 1.0 - cos

        return torch.mean(loss)

class NormalizedSpectralL1Loss(nn.Module):
    """
    归一化光谱方向 L1 损失。
    直接约束每个像素的光谱曲线方向，比 1-cos 更容易产生有效梯度。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1, keepdim=True) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1, keepdim=True) + self.eps)

        pred_dir = pred / (pred_norm + self.eps)
        target_dir = target / (target_norm + self.eps)

        return self.l1(pred_dir, target_dir)

class VisibleSAMLoss(nn.Module):
    """
    可见光区域 SAM 损失。
    默认用于 430-700nm 区域。
    """

    def __init__(self, wavelengths, wl_min=430.0, wl_max=700.0):
        super().__init__()

        import numpy as np

        wavelengths = np.asarray(wavelengths).astype("float32")
        indices = np.where((wavelengths >= wl_min) & (wavelengths <= wl_max))[0]

        if len(indices) == 0:
            raise ValueError("No visible bands found for VisibleSAMLoss.")

        self.register_buffer(
            "indices",
            torch.from_numpy(indices).long(),
        )

        self.sam = SAMLoss()

    def forward(self, pred, target):
        indices = self.indices.to(pred.device)

        pred_visible = pred.index_select(1, indices)
        target_visible = target.index_select(1, indices)

        return self.sam(pred_visible, target_visible)

class SRFRegionSpectralLoss(nn.Module):
    """
    SRF响应区域内的光谱形状损失。

    作用：
    对每个WV2波段对应的HSI响应区域，分别约束该区域内的归一化光谱曲线形状，
    用于解决宽波段SRF只能约束加权平均值、不能约束区域内部光谱曲线的问题。
    """

    def __init__(self, srf_weights=None, threshold_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.eps = eps
        self.l1 = nn.L1Loss()

        if srf_weights is None:
            self.register_buffer("srf_weights", torch.empty(0))
        else:
            self.register_buffer(
                "srf_weights",
                torch.tensor(srf_weights, dtype=torch.float32),
            )

    def normalize_region(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + self.eps)
        return x / (norm + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.srf_weights.numel() == 0:
            return pred.new_tensor(0.0)

        total_loss = pred.new_tensor(0.0)
        valid_count = 0

        for i in range(self.srf_weights.shape[0]):
            weight = self.srf_weights[i]
            threshold = torch.max(weight) * self.threshold_ratio
            indices = torch.nonzero(weight > threshold, as_tuple=False).view(-1)

            if indices.numel() < 2:
                continue

            pred_region = pred.index_select(1, indices)
            target_region = target.index_select(1, indices)

            pred_region = self.normalize_region(pred_region)
            target_region = self.normalize_region(target_region)

            total_loss = total_loss + self.l1(pred_region, target_region)
            valid_count += 1

        if valid_count == 0:
            return pred.new_tensor(0.0)

        return total_loss / valid_count

class BaseHSRLoss(nn.Module):
    """
    基础重建阶段损失。
    加强光谱方向约束，用于降低SAM。
    """

    def __init__(
        self,
        scale_ratio: int,
        n_select_bands: int,
        lambda_l1: float = 1.0,
        lambda_sam: float = 0.2,
        lambda_dc: float = 0.1,
        lambda_sgrad: float = 0.05,
        lambda_sdir: float = 0.2,
        lambda_ns_l1: float = 0.5,
        lambda_srf_region: float = 0.3,
        srf_weights=None,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_sam = lambda_sam
        self.lambda_dc = lambda_dc
        self.lambda_sgrad = lambda_sgrad
        self.lambda_sdir = lambda_sdir
        self.lambda_ns_l1 = lambda_ns_l1

        self.l1 = nn.L1Loss()
        self.sam = SAMLoss()
        self.dc = DataConsistencyLoss(
            scale_ratio=scale_ratio,
            n_select_bands=n_select_bands,
            srf_weights=srf_weights,
        )
        self.sgrad = SpectralGradientLoss()
        self.sdir = SpectralDirectionLoss()
        self.ns_l1 = NormalizedSpectralL1Loss()
        self.lambda_srf_region = lambda_srf_region
        self.srf_region = SRFRegionSpectralLoss(srf_weights=srf_weights)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ):
        loss_l1 = self.l1(pred, gt)
        loss_sam = self.sam(pred, gt)
        loss_dc = self.dc(pred, lr_hsi, hr_msi)
        loss_sgrad = self.sgrad(pred, gt)
        loss_sdir = self.sdir(pred, gt)
        loss_ns_l1 = self.ns_l1(pred, gt)
        loss_srf_region = self.srf_region(pred, gt)

        loss = (
                self.lambda_l1 * loss_l1
                + self.lambda_sam * loss_sam
                + self.lambda_dc * loss_dc
                + self.lambda_sgrad * loss_sgrad
                + self.lambda_sdir * loss_sdir
                + self.lambda_ns_l1 * loss_ns_l1
                + self.lambda_srf_region * loss_srf_region
        )

        loss_dict = {
            "loss": loss.detach(),
            "l1": loss_l1.detach(),
            "sam": loss_sam.detach(),
            "dc": loss_dc.detach(),
            "sgrad": loss_sgrad.detach(),
            "sdir": loss_sdir.detach(),
            "ns_l1": loss_ns_l1.detach(),
            "srf_region": loss_srf_region.detach(),
        }

        return loss, loss_dict


