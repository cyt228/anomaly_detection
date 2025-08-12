import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIM(nn.Module):
    """
    Simplified (single-scale) SSIM for RGB tensors in range [0,1].
    Window size 11 with Gaussian weighting.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._make_window(window_size, sigma))  # 1x1xKxK

    def _make_window(self, window_size, sigma):
        # 1D gaussian (tensor)
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        # Outer product -> 2D, then shape to 1x1xKxK
        window_2d = torch.mm(gauss_1d.unsqueeze(1), gauss_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return window_2d  # CPU buffer; will move to device in forward

    def _ssim(self, img1, img2, C1=0.01**2, C2=0.03**2):
        b, c, h, w = img1.shape
        window = self.window.to(dtype=img1.dtype, device=img1.device).expand(c, 1, self.window_size, self.window_size)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=c)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=c) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        return self._ssim(img1, img2)


class ReconLoss(nn.Module):
    """
    Combined loss: alpha*L1 + beta*(1-SSIM)
    (You can set beta=0 to disable SSIM or alpha=0 for pure SSIM)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.ssim = SSIM()

    def forward(self, pred, target):
        loss = 0.0
        if self.alpha:
            loss = loss + self.alpha * self.l1(pred, target)
        if self.beta:
            ssim_val = self.ssim(pred, target)
            loss = loss + self.beta * (1 - ssim_val)
        return loss


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))
