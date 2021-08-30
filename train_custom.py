import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
import yaml
from easydict import EasyDict as edict

from models.psenet import PSENet
from dataset.polygon import PolygonDataSet
from models.loss.psenet_loss import PSENet_Loss
from utils import AverageMeter

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
cuda = torch.cuda.is_available()

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

def train(train_loader, model, model_loss, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()
    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(cfg=cfg))

        imgs = data['imgs']
        gt_texts = data['gt_texts']
        gt_kernels = data['gt_kernels']
        training_masks = data['training_masks']

        if cuda:
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
        loss = 0.7 * loss_text + 0.3 * loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        losses.update(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            output_log = f"{time.asctime(time.localtime())} ({iter + 1:4d}/{len(train_loader):4d}) " \
                         f"LR: {optimizer.param_groups[0]['lr']:.6f} | Batch: {batch_time.avg:.3f}s " \
                         f"Loss: {loss.avg:.3f} | Loss(text/kernel): ({losses_text:.3f}/{losses_kernels:.3f}) " \
                         f"IoU(text/kernel): ({ious_text.avg:.3f}/{ious_kernel.avg:.3f})"
            print(output_log)
            sys.stdout.flush()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train.schedule
    if isinstance(schedule, str):
        assert schedule == 'poly lr', 'Error: schedule should be poly lr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train.epoch * len(dataloader)
        lr = cfg.train.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, list):
        lr = cfg.train.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path):
    if state['epoch'] % 50 == 0:
        ckpt_name = f"psenet_{state['epoch']}ep.pt"
        file_path = os.path.join(checkpoint_path, ckpt_name)
        torch.save(state, file_path)


def main(opt):
    cfg = edict(yaml.safe_load(open(opt.cfg)))
    # model
    model = PSENet(**cfg.model)
    model_loss = PSENet_Loss(**cfg.loss)

    # data loader
    data_loader = PolygonDataSet('train')
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    workspace = os.path.join(opt.project, opt.name)
    store_dir = os.path.join(workspace, 'ckpt')
    
    start_epoch = 0
    start_iter = 0
    if opt.pretrain:
        pretrain_file = opt.weights
        assert os.isfile(pretrain_file), 'Error: no pretrained weights found!'
        checkpoint = torch.load(pretrain_file)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Finetuning from pretrained model {color_str(pretrain_file)}')
    elif opt.resume:
        if not os.path.exists(restore_path):
            print(f"No restore file: {color_str(restore_path, 'red')}")
            exit()
        ckpt = glob.glob(os.path.join(store_dir, "*.pt"))
        if len(ckpt) == 0:
            print(f"There is No Files in {color_str(store_dir, 'red')}")
            exit()
        ckpt.sort(key=lambda x: os.path.getmtime(x))
        restore_path = ckpt[-1]
        checkpoint = torch.load(restore_path)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"restore from {color_str(restore_path)}")
    else:
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        print(f"Train model from scratch and save at {color_str(store_dir)}")

    for epoch in range(start_epoch, opt.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, opt.epoch))

        train(train_loader, model, model_loss, optimizer, epoch, start_iter, cfg)

        state = dict(
            epoch=epoch,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        save_checkpoint(state, store_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/psev1.yaml')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--name', type=str, default='v1.0.0')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--weights', type=str, default='epoch.pt')
    parser.add_argument('--resume', action='store_true')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
