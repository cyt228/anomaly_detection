
import os
import sys
from typing import Literal
import torch
import torch.nn as nn

# Try to locate UNet-family next to this package or via env var UNET_FAMILY_PATH
def _ensure_unet_family_on_path():
    candidates = []
    # 1) Env var
    p = os.environ.get("UNET_FAMILY_PATH")
    if p:
        candidates.append(p)
    # 2) Sibling folder
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "UNet-family"))
    candidates.append(os.path.join(os.path.dirname(here), "UNet-family"))
    candidates.append(os.path.join(os.getcwd(), "UNet-family"))
    for c in candidates:
        if c and os.path.isdir(c):
            if c not in sys.path:
                sys.path.insert(0, c)
            # network files live at UNet-family/networks
            networks = os.path.join(c, "networks")
            if os.path.isdir(networks) and networks not in sys.path:
                sys.path.insert(0, networks)
            return
    # If we reach here, we couldn't find it; leave import to fail with a clear message.
_ensure_unet_family_on_path()

try:
    from networks.UNet import UNet
    from networks.UNet_Nested import UNet_Nested
except Exception as e:
    raise ImportError(
        "Could not import UNet modules. Make sure the 'UNet-family' folder (with 'networks' inside) "
        "is placed next to this script or set environment variable UNET_FAMILY_PATH to its directory.\n"
        f"Original error: {e}"
    )


class UNetRecon(nn.Module):
    """
    Wraps UNet/UNet++ to perform RGB image reconstruction.
    - input: Bx3xHxW in [0,1]
    - output: Bx3xHxW in [0,1] (via optional sigmoid clamp)
    """
    def __init__(
        self,
        backbone: Literal["unet", "unetpp"] = "unet",
        feature_scale: int = 1,
        is_deconv: bool = True,
        is_batchnorm: bool = True,
        out_activation: Literal["sigmoid", "none"] = "sigmoid",
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()
        if backbone.lower() == "unet":
            self.backbone = UNet(in_channels=in_channels, n_classes=out_channels,
                                 feature_scale=feature_scale, is_deconv=is_deconv, is_batchnorm=is_batchnorm)
        
        elif backbone.lower() in ("unetpp", "unet++"):
            self.backbone = UNet_Nested(in_channels=in_channels, n_classes=out_channels,
                                        feature_scale=feature_scale, is_deconv=is_deconv,
                                        is_batchnorm=is_batchnorm, is_ds=False)
        else:
            raise ValueError("backbone must be one of: 'unet', 'unetpp'")
        self.out_activation = out_activation

    def forward(self, x):
        y = self.backbone(x)
        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        return y
