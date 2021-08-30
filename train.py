import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset.polygon import PolygonDataSet
from models import build_model
from utils import AverageMeter

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)


def train(train_loader, model, optimizer, epoch, start_iter, cfg):
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

        # forward
        outputs = model(**data)
        #
        # print(outputs['loss_text'].shape)
        # print(outputs['loss_kernels'].shape)

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
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel): {loss_text:.3f}/{loss_kernel:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} '.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.param_groups[0]['lr'],
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernels.avg,
                loss=losses.avg,
                iou_text=ious_text.avg,
                iou_kernel=ious_kernel.avg,
                # acc_rec=accs_rec.avg,
            )
            print(output_log)
            sys.stdout.flush()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule
    if isinstance(schedule, str):
        assert schedule == 'poly lr', 'Error: schedule should be poly lr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if cfg.data.train.type in ['synth'] or (state['iter'] == 0 and state['epoch'] % 50 == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)


def main(args):
    cuda = torch.cuda.is_available()
    cfg = Config.fromfile(args.config)
    print(cfg)
    exit()
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('/data/weixianwei/psenet/models/', cfg_name)

    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()

    # data loader
    data_loader = PolygonDataSet('train', )
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    # model
    model = build_model(cfg.model)
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_cfg.lr)

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume:
        # todo 断点训练
        restore_path = args.resume
        if not os.path.exists(restore_path):
            print("there is no restore file: ", restore_path)
            exit()
        print(f"restore from {restore_path}!")
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, epoch, start_iter, cfg)

        state = dict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        save_checkpoint(state, checkpoint_path, cfg)

def parse_opt():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', default="uniform_v1.0.0.py", help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str,
                        default="checkpoints/uniform_500ep.pth.tar")
    args = parser.parse_args()

    main(args)
