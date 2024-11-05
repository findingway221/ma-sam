from math import ceil

import torch
import torch.distributed as dist
from metric.myMetric import multiclass_dice_coeff

def train_with_atlas_DDP_2D(rank, net, monitor, sampler, dataloader, loss_function, optimizer, epoch, device, args):
    def init():
        loss_num = torch.tensor([0], device=device, dtype=torch.float32)
        total = torch.tensor([0], device=device, dtype=torch.int32)
        return loss_num, total

    interval = ceil(250 / args.train_batch_size)

    net.train()
    sampler.set_epoch(epoch)

    if rank == 0:
        pbar = monitor.train_epoch_start(epoch=epoch)

    loss_num, total = init()

    dist.barrier()
    for step, (img_array, gt_array, atlas_mask_array) in enumerate(dataloader):
        img = img_array.to(device, torch.float32)
        atlas_mask = atlas_mask_array.to(device, torch.float32)
        gt = gt_array.to(device, torch.int32)

        predict_logits = net(img, atlas_mask)

        loss = loss_function(predict_logits, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_num += loss.detach()
        total += 1

        if (step+1) % interval == 0 or (step+1) == len(dataloader):
            updates = total.item()
            dist.reduce(loss_num, 0, op=dist.ReduceOp.SUM)
            dist.reduce(total, 0, op=dist.ReduceOp.SUM)

            if rank == 0:
                monitor.train_step_end(epoch=epoch, step=step, loss=(loss_num/total).item(), pbar=pbar,
                                       updates=updates)

            loss_num, total = init()


def val_with_atlas_DDP_2D(rank, net, monitor, dataloader, loss_function, optimizer, epoch, device, args, image_size=512):
    def init():
        pred_3D = torch.empty((0, 1, image_size, image_size), dtype=torch.int32, device=device)
        gt_3D = torch.empty((0, 1, image_size, image_size), dtype=torch.int32, device=device)
        loss_num = torch.tensor([0], device=device, dtype=torch.float32)
        total = torch.tensor([0], device=device, dtype=torch.int32)
        return pred_3D, gt_3D, loss_num, total

    interval = ceil(250/args.val_batch_size)

    net.eval()

    if rank == 0:
        pbar = monitor.val_epoch_start(epoch=epoch, device=device)

    with torch.no_grad():
        dist.barrier()

        pred_3D, gt_3D, loss_num, total = init()
        for step, (img_array, gt_array, atlas_mask_array) in enumerate(dataloader):
            img = img_array.to(device, torch.float32)
            atlas_mask = atlas_mask_array.to(device, torch.float32)
            gt = gt_array.to(device, torch.int32)

            predict_logits = net(img, atlas_mask)

            loss = loss_function(predict_logits, gt)

            loss_num += loss.detach()
            total += 1

            output_softmax = torch.softmax(predict_logits, 1)
            output = torch.argmax(output_softmax, dim=1, keepdim=True)
            pred_3D = torch.cat([pred_3D, output], dim=0)
            gt_3D = torch.cat([gt_3D, gt], dim=0)

            if (step+1) % interval == 0 or (step+1) == len(dataloader):
                dice_list = multiclass_dice_coeff(pred=pred_3D, target=gt_3D, class_num=args.num_classes,
                                                  reduction='None', need_softmax=False, need_argmax=False)
                dice_tensor = torch.tensor(dice_list, device=device)

                updates = total.item()
                dist.reduce(loss_num, 0, op=dist.ReduceOp.SUM)
                dist.reduce(total, 0, op=dist.ReduceOp.SUM)
                dist.reduce(dice_tensor, 0, op=dist.ReduceOp.SUM)

                if rank == 0:
                    monitor.val_step_end(epoch=epoch, step=step, dice_tensor=(dice_tensor/args.world_size),
                                     loss=(loss_num/total).item(), pbar=pbar, updates=updates)

                pred_3D, gt_3D, loss_num, total = init()


        if rank == 0:
            times = len(dataloader) // interval + (1 if len(dataloader) % interval != 0 else 0)
            monitor.val_epoch_end(epoch=epoch, net=net, optimizer=optimizer, times=times)


def train_DDP_2D(rank, net, monitor, sampler, dataloader, loss_function, optimizer, epoch, device, args):
    def init():
        loss_num = torch.tensor([0], device=device, dtype=torch.float32)
        total = torch.tensor([0], device=device, dtype=torch.int32)
        return loss_num, total

    interval = ceil(250 / args.train_batch_size)

    net.train()
    sampler.set_epoch(epoch)

    if rank == 0:
        pbar = monitor.train_epoch_start(epoch=epoch)

    loss_num, total = init()

    dist.barrier()
    for step, (img_array, gt_array) in enumerate(dataloader):
        img = img_array.to(device, torch.float32)
        gt = gt_array.to(device, torch.int32)

        predict_logits = net(img)

        loss = loss_function(predict_logits, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_num += loss.detach()
        total += 1

        if (step+1) % interval == 0 or (step+1) == len(dataloader):
            updates = total.item()
            dist.reduce(loss_num, 0, op=dist.ReduceOp.SUM)
            dist.reduce(total, 0, op=dist.ReduceOp.SUM)

            if rank == 0:
                monitor.train_step_end(epoch=epoch, step=step, loss=(loss_num/total).item(), pbar=pbar,
                                       updates=updates)

            loss_num, total = init()


def val_DDP_2D(rank, net, monitor, dataloader, loss_function, optimizer, epoch, device, args, image_size):
    def init():
        pred_3D = torch.empty((0, 1, image_size, image_size), dtype=torch.int32, device=device)
        gt_3D = torch.empty((0, 1, image_size, image_size), dtype=torch.int32, device=device)
        loss_num = torch.tensor([0], device=device, dtype=torch.float32)
        total = torch.tensor([0], device=device, dtype=torch.int32)
        return pred_3D, gt_3D, loss_num, total

    interval = ceil(250/args.val_batch_size)

    net.eval()

    if rank == 0:
        pbar = monitor.val_epoch_start(epoch=epoch, device=device)

    with torch.no_grad():
        dist.barrier()

        pred_3D, gt_3D, loss_num, total = init()
        for step, (img_array, gt_array) in enumerate(dataloader):
            img = img_array.to(device, torch.float32)
            gt = gt_array.to(device, torch.int32)

            predict_logits = net(img)

            loss = loss_function(predict_logits, gt)

            loss_num += loss.detach()
            total += 1

            output_softmax = torch.softmax(predict_logits, 1)
            output = torch.argmax(output_softmax, dim=1, keepdim=True)
            pred_3D = torch.cat([pred_3D, output], dim=0)
            gt_3D = torch.cat([gt_3D, gt], dim=0)

            if (step+1) % interval == 0 or (step+1) == len(dataloader):
                dice_list = multiclass_dice_coeff(pred=pred_3D, target=gt_3D, class_num=args.num_classes,
                                                  reduction='None', need_softmax=False, need_argmax=False)
                dice_tensor = torch.tensor(dice_list, device=device)

                updates = total.item()
                dist.reduce(loss_num, 0, op=dist.ReduceOp.SUM)
                dist.reduce(total, 0, op=dist.ReduceOp.SUM)
                dist.reduce(dice_tensor, 0, op=dist.ReduceOp.SUM)

                if rank == 0:
                    monitor.val_step_end(epoch=epoch, step=step, dice_tensor=(dice_tensor/args.world_size),
                                     loss=(loss_num/total).item(), pbar=pbar, updates=updates)

                pred_3D, gt_3D, loss_num, total = init()


        if rank == 0:
            times = len(dataloader) // interval + (1 if len(dataloader) % interval != 0 else 0)
            monitor.val_epoch_end(epoch=epoch, net=net, optimizer=optimizer, times=times)
