import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PSENet(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)
        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs, img_metas=None, cfg=None):
        start = time.time()

        # backbone
        f = self.backbone(imgs)

        # FPN
        f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)

        # detection

        det_out = self.det_head(f)

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            det_out = self.det_head.get_results(det_out, img_metas, cfg)
        return det_out
