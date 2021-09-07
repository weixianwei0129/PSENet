import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


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
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=False)

    def forward(self, imgs):
        # backbone
        f = self.backbone(imgs)

        # FPN
        f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)

        # detection

        det_out = self.det_head(f)
        det_out = self._upsample(det_out, imgs.size())
        return det_out


if __name__ == '__main__':
    import yaml
    import numpy as np
    from easydict import EasyDict
    from torchsummary import summary

    x = torch.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.float32))
    cfg = EasyDict(yaml.safe_load(open('../config/pse_v1.3.0.yaml')))
    model = PSENet(**cfg.model)
    # summary(model, (3, 320, 320))
    x = model(x)
    print(x.shape)
