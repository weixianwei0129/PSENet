import torch
import torch.nn as nn
from ..loss import build_loss, ohem_batch, iou


class PSENet_Loss(nn.Module):
    def __init__(self,
                 loss_text,
                 loss_kernel):
        super(PSENet_Loss, self).__init__()

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)

    def forward(self, out, gt_texts, gt_kernels, training_masks):
        # text loss
        texts = out[:, 0, :, :]
        selected_masks = ohem_batch(texts, gt_texts, training_masks)

        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)  # 实现论文公式（6），分类loss
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)  # 计算iou
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss 论文的公式（7）
        kernels = out[:, 1:, :, :]
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        return losses
