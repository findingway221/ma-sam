import json
import logging
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Monitor:
    def __init__(self, weight_base_path, resume_train, class_num, best_dice, best_epoch, train_length, val_length,
                 early_stop=False, save_every=1):
        self.weight_base_path = weight_base_path
        self.class_num = class_num
        self.best_dice = best_dice
        self.best_epoch = best_epoch
        self.train_length = train_length
        self.val_length = val_length
        self.early_stop = early_stop
        self.save_every = save_every

        writerPath = f'{weight_base_path}/logs'
        logPath = f'{weight_base_path}/logs/log.log'
        self.jsonPath = f'{weight_base_path}/summary.json'
        self.time_csv_path = f'{weight_base_path}/time.csv'

        if self.early_stop:
            self.early_stop_flag = 0
            self.early_stop_patience = 10

        if not resume_train:
            if os.path.exists(writerPath):
                shutil.rmtree(writerPath)

        os.makedirs(writerPath, exist_ok=True)
        self.writer = SummaryWriter(log_dir=writerPath)
        logging.basicConfig(format='%(message)s', filename=logPath, level=logging.INFO)




    def info_to_file(self, epoch, args):
        with open(self.jsonPath, 'w', encoding='utf-8') as f:
            json.dump({'bestDice': self.best_dice, 'bestEpoch': self.best_epoch,
                       'nowEpoch': epoch, 'args': args}, f)


    def train_epoch_start(self, epoch):
        logging.info(f'=======================train epoch:{epoch}========================')
        pbar = tqdm(total=self.train_length, ncols=80)
        pbar.set_description(f'train--epoch {epoch}')

        return pbar


    def train_step_end(self, epoch, step, loss, pbar, updates=1):
        self.writer.add_scalar('train_loss', loss, epoch * self.train_length + step)
        self.writer.flush()
        logging.info(f'epoch:{epoch}\tstep:{step}\tloss:{loss}')

        pbar.set_postfix(**{'loss': loss})
        pbar.update(updates)

    def val_epoch_start(self, epoch, device):
        self.diceMean = torch.tensor([0], dtype=torch.float, device=device)
        self.diceClass = torch.zeros((self.class_num), dtype=torch.float, device=device)

        logging.info(f'=======================val epoch:{epoch}========================')
        pbar = tqdm(total=self.val_length, ncols=80)
        pbar.set_description(f'val--epoch {epoch}')

        return pbar



    def val_step_end(self, epoch, step, dice_tensor, loss, pbar, updates=1):
        meanDice = torch.mean(dice_tensor[1:]).item()

        self.diceMean += meanDice
        self.diceClass += dice_tensor

        message = f'meandice:{meanDice:.5f}\t'
        for i in range(self.class_num):
            message = message + f'dice{i}:{dice_tensor[i].item():.5f}\t'
        logging.info(f'epoch{epoch}:\t' + message)

        self.writer.add_scalar('val_loss', loss, epoch * self.val_length + step)
        self.writer.add_scalar('val_dice', meanDice, epoch * self.val_length + step)
        self.writer.flush()

        pbar.set_postfix(**{'dice': meanDice})
        pbar.update(updates)


    def val_epoch_end(self, epoch, net, optimizer, times):
        self.diceMean = self.diceMean / times
        self.diceClass = self.diceClass / times

        meanDice = self.diceMean.item()
        meanClassDice = self.diceClass.tolist()

        message = f'meandice:{meanDice:.5f}\t'
        for i in range(self.class_num):
            message = message + f'dice{i}:{meanClassDice[i]:.5f}\t'
        logging.info(message)

        self.save_state_dict(epoch, meanDice, net, optimizer)

        self.writer.add_scalar('val_dice_epoch', meanDice, epoch)
        self.writer.flush()

        print(f'dice:{meanDice:.5f}')


    def save_state_dict(self, epoch, dice, net, optimizer):
        if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        if dice > self.best_dice:
            self.best_dice = dice
            self.best_epoch = epoch
            dicBest = {
                'epoch': epoch,
                'best_epoch': self.best_epoch,
                'best_dice': self.best_dice,
                'model': state_dict,
                'optimizer': optimizer.state_dict()
            }

            bestPath = self.weight_base_path + os.sep + 'best.pth'

            torch.save(dicBest, bestPath)

            logging.info(f'save best model! dice:{dice}')
            print(f'save best model! dice:{dice}')

            if self.early_stop:
                self.early_stop_flag = -1

        if epoch == 0 or (epoch+1) % self.save_every == 0:
            dicLatest = {
                'epoch': epoch,
                'best_epoch': self.best_epoch,
                'best_dice': self.best_dice,
                'model': state_dict,
                'optimizer': optimizer.state_dict()
            }
            latestPath = self.weight_base_path + os.sep + 'latest.pth'
            torch.save(dicLatest, latestPath)

    def early_stop_step(self):
        if(self.early_stop):
            self.early_stop_flag += 1

            if self.early_stop_flag >= self.early_stop_patience:
                return True

        else:
            return False


    def end(self):
        print(f'bestDice: {self.best_dice}, achieved by epoch: {self.best_epoch}')