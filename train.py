import os
import time
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter
from models.psenet import PSENet
from dataset.polygon import PolygonDataSet
from models.loss.psenet_loss import PSENet_Loss
from models.post_processing.tools import get_results

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


def norm_img(img):
    return (img - img.min()) / (img.max() - img.min())


def check_file(cfg, opt):
    assert cfg.model.detection_head.kernel_num == cfg.data.kernel_num
    assert opt.name in opt.cfg


def one_cycle(y1=1.0, y2=0.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - np.cos(x * np.pi / steps)) / 2) * (y2 - y1) + y1


def color_str(string, color='blue'):
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return f"{colors.get(color, colors['red'])}{string}{colors['end']}"


def concat_img(imgs, gt_texts, gt_kernels):
    idx = np.random.randint(0, imgs.shape[0])
    concat = [norm_img(imgs[idx]), torch.stack([norm_img(gt_texts[idx])] * 3, dim=0)]
    for i in range(gt_kernels.shape[1]):
        concat.append(torch.stack([norm_img(gt_kernels[idx, i, ...])] * 3, dim=0))
    return torch.cat(concat, dim=2)


@torch.no_grad()
def test(test_loader, model, model_loss, epoch, cfg, writer):
    model.eval()
    total_data_num = len(test_loader)
    # meters
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()

    for iter, data in tqdm(enumerate(test_loader)):

        imgs = data['imgs']
        gt_texts = data['gt_texts']
        gt_kernels = data['gt_kernels']
        training_masks = data['training_masks']
        if cuda == 'cuda':
            imgs = imgs.cuda()
            gt_texts = gt_texts.cuda()
            gt_kernels = gt_kernels.cuda()
            training_masks = training_masks.cuda()

        # forward
        out = model(imgs=imgs)
        outputs = model_loss(out, gt_texts, gt_kernels, training_masks)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())

        # 论文公式（4）
        loss = cfg.loss.loss_text.loss_weight * loss_text + \
               cfg.loss.loss_kernel.loss_weight * loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        losses.update(loss.item())

        if iter % np.ceil(total_data_num / 5) == 0:
            score, label = get_results(out, cfg.evaluation.kernel_num, cfg.evaluation.min_area)
            score = torch.from_numpy(score)
            label = torch.from_numpy(label)
            if cuda == 'cuda':
                score = score.cuda()
                label = label.cuda()
            concat = [norm_img(imgs[0]), torch.stack([norm_img(gt_texts[0])] * 3, dim=0)]
            concat.append(concat[0] * score[None, ...])
            concat.append(torch.stack([norm_img(label)] * 3, dim=0))
            concat = torch.cat(concat, dim=2)
            writer.add_image(f'test/img/{iter}', concat, epoch)

    # ====Summery====
    writer.add_scalar('test/loss/total loss', losses.avg, epoch)
    writer.add_scalar('test/loss/text loss', losses_text.avg, epoch)
    writer.add_scalar('test/loss/kernel loss', losses_kernels.avg, epoch)
    writer.add_scalar('test/IoU/text IoU', ious_text.avg, epoch)
    writer.add_scalar('test/IoU/kernel IoU', ious_kernel.avg, epoch)
    output_log = f"[TEST] {time.asctime(time.localtime())} " \
                 f"Loss: {losses.avg:.3f} | " \
                 f"Loss(text/kernel): ({losses_text.avg:.3f}/{losses_kernels.avg:.3f}) " \
                 f"IoU(text/kernel): ({ious_text.avg:.3f}/{ious_kernel.avg:.3f})\n\n"
    print(color_str(output_log, 'red'))
    return losses.avg


def train(train_loader, model, model_loss, optimizer, epoch, scaler, cfg, writer):
    model.train()
    total_data_num = len(train_loader)
    # meters
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()

    # 增大batch_size
    train_batch_size = cfg.train.batch_size
    data_batch_size = cfg.data.batch_size
    cur_batch_size = data_batch_size

    optimizer.zero_grad()
    for iter, data in enumerate(train_loader):

        # prepare input
        data.update(dict(cfg=cfg))

        imgs = data['imgs']
        gt_texts = data['gt_texts']
        gt_kernels = data['gt_kernels']
        training_masks = data['training_masks']
        if cuda == 'cuda':
            imgs = imgs.cuda()
            gt_texts = gt_texts.cuda()
            gt_kernels = gt_kernels.cuda()
            training_masks = training_masks.cuda()

        # forward
        det_out = model(imgs=imgs)
        outputs = model_loss(det_out, gt_texts, gt_kernels, training_masks)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())

        # 论文公式（4）
        loss = cfg.loss.loss_text.loss_weight * loss_text + \
               cfg.loss.loss_kernel.loss_weight * loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        losses.update(loss.item())

        # backward
        scaler.scale(loss).backward()
        if cur_batch_size >= train_batch_size or \
                iter == len(train_loader) - 1:
            scaler.step(optimizer)  # optimizer.step()
            scaler.update()
            optimizer.zero_grad()
            cur_batch_size = data_batch_size
        else:
            cur_batch_size += data_batch_size

        # print log
        if iter % np.ceil(total_data_num / 5) == 0:
            step = epoch * len(train_loader) + iter
            # ====Summery====
            writer.add_image('train/img', concat_img(imgs, gt_texts, gt_kernels), step)
            writer.add_scalar('train/loss/total loss', losses.avg, step)
            writer.add_scalar('train/loss/text loss', losses_text.avg, step)
            writer.add_scalar('train/loss/kernel loss', losses_kernels.avg, step)
            writer.add_scalar('train/IoU/text IoU', ious_text.avg, step)
            writer.add_scalar('train/IoU/kernel IoU', ious_kernel.avg, step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
            output_log = f"[{time.asctime(time.localtime())}] " \
                         f"[{iter + 1:4d}/{len(train_loader):4d}] " \
                         f"LR: {optimizer.param_groups[0]['lr']:.6f} | " \
                         f"Loss(total/text/kernel): " \
                         f"({losses.avg:.3f}/{losses_text.avg:.3f}/{losses_kernels.avg:.3f}) | " \
                         f"IoU(text/kernel): ({ious_text.avg:.3f}/{ious_kernel.avg:.3f})"
            print(output_log)


def save_checkpoint(state, checkpoint_path):
    if state['epoch'] % 50 == 0:
        ckpt_name = f"psenet_{state['epoch']}ep.pt"
        file_path = os.path.join(checkpoint_path, ckpt_name)
        torch.save(state, file_path)


def main(opt):
    cfg = EasyDict(yaml.safe_load(open(opt.cfg)))
    # model
    model = PSENet(**cfg.model)
    model_loss = PSENet_Loss(**cfg.loss)

    # data loader
    train_dataset = PolygonDataSet('train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=4,  # 可用GPU数量的四倍
        drop_last=True,
        pin_memory=True
    )

    test_dataset = PolygonDataSet('test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=3
    )

    if cuda == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # 创建优化器
    # Check if model has custom optimizer / loss
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train.optimizer == 'SGD':
            optimizer = SGD(g0, lr=cfg.train.lr, momentum=0.99, nesterov=True)
        elif cfg.train.optimizer == 'Adam':
            optimizer = Adam(g0, lr=cfg.train.lr)
        else:
            raise Exception(f"Error optimizer method! ({cfg.train.optimizer})")
    optimizer.add_param_group({'params': g1, 'weight_decay': 5e-4})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    # Specifying the disk address
    check_file(cfg, opt)
    workspace = os.path.join(opt.project, opt.name)
    store_dir = os.path.join(workspace, 'ckpt')
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
        print(f"create a path {color_str(store_dir)}.")

    writer = SummaryWriter(log_dir=workspace, flush_secs=30)

    # select train type :
    # 1. load pretrain model; 2. resume train; 3. train from scratch
    start_epoch = 0
    best_loss = np.inf
    if opt.pretrain:
        pretrain_file = opt.weights
        assert os.path.isfile(pretrain_file), \
            'Error: no pretrained weights found!'
        checkpoint = torch.load(pretrain_file)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Fine tuning from {color_str('pretrained model', 'red')} {color_str(pretrain_file)}")
    elif opt.resume:
        checkpoint_path = os.path.join(store_dir, "last.pt")
        assert os.path.isfile(checkpoint_path), \
            f'Error: no checkpoint file found at {color_str(checkpoint_path)}'

        checkpoint = torch.load(checkpoint_path)
        if not opt.force:
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', np.inf)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"{color_str('restore', 'red')} from {color_str(checkpoint_path)}")
    else:
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        print(f"Train model from {color_str('scratch', 'red')} and save at {color_str(store_dir)}")

    # Scheduler 设置
    if opt.linear_lr:
        lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - cfg.train.lf) + cfg.train.lf  # linear
    else:
        lf = one_cycle(1, cfg.train.lf, opt.epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # 混合精度 FP32 + FP16
    scaler = amp.GradScaler(enabled=cuda)

    # Loop all train data
    for epoch in range(start_epoch, opt.epochs):

        # warmup
        if opt.warmup and epoch <= cfg.train.warmup_epochs:
            xi = [0, cfg.train.warmup_epochs]
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(epoch, xi, [0.0, x['initial_lr'] * lf(epoch)])

        t1 = time.time()
        print(f"\nEpoch: ({epoch + 1:4d}/{opt.epochs:4d})")
        train(train_loader, model, model_loss, optimizer, epoch, scaler, cfg, writer)
        scheduler.step()
        print(f"An epoch takes {time.time() - t1:.3f}s")

        # save model and optimizer
        if epoch % 50 == 0:
            state = dict(
                epoch=epoch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
            torch.save(state, os.path.join(store_dir, 'last.pt'))
            if epoch > opt.epochs * .1:
                test_loss = test(test_loader, model, model_loss, epoch, cfg, writer)
                if test_loss < best_loss:
                    state.update(test_loss=test_loss)
                    torch.save(state, os.path.join(store_dir, 'best.pt'))
                    best_loss = test_loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/xxx.yaml', help="Description from README.md.")
    parser.add_argument('--linear_lr', action='store_true', help="If true, use linear lr, else use cosine lr")
    parser.add_argument('--epochs', type=int, default=300, help='Total epoch during training.')
    parser.add_argument('--project', type=str, default='', help='Project path on disk')
    parser.add_argument('--name', type=str, default='vx.x.x', help='Name of train model')
    parser.add_argument('--pretrain', action='store_true', help='Whether to use a pre-training model')
    parser.add_argument('--weights', type=str, default='xx.pt', help="Pretrain the model's path on disk")
    parser.add_argument('--resume', action='store_true',
                        help="Whether to resume, and find the `last.pt` file as weights")
    parser.add_argument('--warmup', action='store_true', help="Warmup learning rate")
    parser.add_argument('--force', action='store_true',
                        help="If True, only reload weights and optimizer, else reload epoch number and test loss")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
