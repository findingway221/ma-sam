import argparse
import os
from types import SimpleNamespace

import torch

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ..model.segmentation_sub_network.MASAM import MASAM, load_from, froze
from ..model.utils import set_seed
from ..monitor.Monitor import Monitor

from ..loss_fun.CEDCLoss import CEDCLoss

from ..dataset.spine_dataset import SpineDataset
from trainer.Trainer import val_with_atlas_DDP_2D, train_with_atlas_DDP_2D

from utils.properties import properties

import torch.distributed as dist
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9999'


def process(rank, args, train_dataset, val_dataset, weight_base_path, sam_weight_path, config):

    set_seed(221)
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=args.world_size)
    device = torch.device(f'cuda:{rank}')

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, num_workers=args.train_workers,
                                  batch_size=args.train_batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=args.val_workers,
                                batch_size=args.val_batch_size)



    net = MASAM(args.num_classes, args.num_atlases,
                prompt_embed_dim=config.prompt_embed_dim, image_size=config.image_size,
                vit_patch_size=config.vit_patch_size, encoder_embed_dim=config.encoder_embed_dim,
                encoder_depth=config.encoder_depth, encoder_num_heads=config.encoder_num_heads,
                encoder_global_attn_indexes=config.encoder_global_attn_indexes, adapterTrain=args.adapter_train)

    epoch_start, best_dice, best_epoch, model_dic, opt_dic = 0, 0, 0, None, None
    if args.resume_train:
        pth_name = 'latest.pth'
        weight_path = weight_base_path + os.sep + pth_name
        if os.path.exists(weight_path):
            with open(weight_path, "rb") as f:
                dic = torch.load(f)
                epoch_start, best_dice, best_epoch, model_dic, opt_dic = dic['epoch'] + 1, dic['best_dice'], \
                    dic['best_epoch'], dic['model'], dic['optimizer']
                net.load_state_dict(model_dic)
            if rank==0:
                print(f'load {pth_name}, exiting best dice: {best_dice}, achieved by epoch: {best_epoch}')
        else:
            if rank==0:
                print(f'err! weight not exit! use path {weight_path}')
            exit()
    else:
        with open(sam_weight_path, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')

        if rank==0: print('interpolate sam-pretrained weight')
        net_new_dict = load_from(net.state_dict(), state_dict, config.image_size, config.vit_patch_size)
        net.load_state_dict(net_new_dict)
        if rank==0: print(f"load sam-pretrained weight: {sam_weight_path.split('/')[-1]}")


    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    torch.cuda.empty_cache()

    if rank == 0:
        monitor = Monitor(weight_base_path=weight_base_path, resume_train=args.resume_train, class_num=args.num_classes,
                        best_dice=best_dice, best_epoch=best_epoch, train_length=len(train_dataloader),
                        val_length=len(val_dataloader), save_every=args.save_every)
    else:
        monitor=None

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    if opt_dic is not None:
        optimizer.load_state_dict(opt_dic)

    cedcloss = CEDCLoss(args.num_classes).to(device)

    early_stop_flag = torch.tensor(0, device=device)
    for epoch in range(epoch_start, args.epochs):
        froze(net)
        train_with_atlas_DDP_2D(rank, net, monitor, train_sampler, train_dataloader, cedcloss, optimizer, epoch, device, args)

        val_with_atlas_DDP_2D(rank, net, monitor, val_dataloader, cedcloss, optimizer, epoch, device, args, image_size)

        if rank == 0:
            monitor.info_to_file(epoch=epoch, args=args.__dict__)
            if monitor.early_stop_step():
                early_stop_flag = torch.tensor(1, device=device)

        dist.barrier()
        dist.all_reduce(early_stop_flag, op=dist.ReduceOp.MAX)
        if early_stop_flag.item() > 0: break

    if rank == 0:
        monitor.end()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--num_classes', type=int, default=4, help='class num')
    parser.add_argument('--num_atlases', type=int, default=20, help='atlas num')
    parser.add_argument('--size', type=str, default='small', help='size of sam image encoder')
    parser.add_argument('--train_batch_size', type=int, default=4, help='train batch size for each gpu')
    parser.add_argument('--val_batch_size', type=int, default=32, help='val batch size for each gpu')
    parser.add_argument('--train_workers', type=int, default=2, help='train worker num')
    parser.add_argument('--val_workers', type=int, default=2, help='val worker num')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save_every', type=int, default=5, help='how many epochs to save model')
    parser.add_argument('--early_stop', action='store_true', help='early stop')
    parser.add_argument('--resume_train', action='store_true', help='resume train')
    parser.add_argument('--adapter_train', action='store_true', help='use adapter')
    args = parser.parse_args()
    print(args.__dict__)

    data = ''   # todo: remember to configure the properties.py

    image_size = 512
    args.world_size = torch.cuda.device_count()
    print(f'num of gpu: {torch.cuda.device_count()} ')

    weight_base_path = properties[data]['weight_path'] + f'/masam'
    os.makedirs(weight_base_path, exist_ok=True)

    img_base_path = properties[data]['img_path']
    mask_base_path = properties[data]['mask_path']
    atlas_base_path = properties[data]['atlas_path']
    data_file_path = properties[data]['data_file_path']

    sam_weight_base_path = properties['SAM_weight_path']
    sam_weight_name = properties['model_size'][args.size]['name']
    sam_weight_path = f'{sam_weight_base_path}/{sam_weight_name}'

    train_img_file = data_file_path + '/trainImg.txt'
    val_img_file = data_file_path + '/valImg.txt'
    train_mask_file = data_file_path + '/trainMask.txt'
    val_mask_file = data_file_path + '/valMask.txt'

    train_dataset = SpineDataset()  # todo: remember to configure your dataset
    val_dataset = SpineDataset()    # todo: remember to configure your dataset

    config = SimpleNamespace(
        prompt_embed_dim = 256,
        vit_patch_size = 16,
        image_size = image_size,
        encoder_embed_dim = properties['modelSize'][args.size]['encoder_embed_dim'],
        encoder_depth = properties['modelSize'][args.size]['encoder_depth'],
        encoder_num_heads = properties['modelSize'][args.size]['encoder_num_heads'],
        encoder_global_attn_indexes = properties['modelSize'][args.size]['encoder_global_attn_indexes'])

    mp.spawn(process, args=(args, train_dataset, val_dataset, weight_base_path, sam_weight_path, config, ),
             nprocs=args.world_size, join=True)
