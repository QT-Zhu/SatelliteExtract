from torch.utils.data.dataloader import DataLoader
from utils.AerialDataset import AerialDataset
import torch
import os
import torch.nn as nn
import torch.optim as opt
from utils.utils import ret2mask,get_test_times
import matplotlib.pyplot as plt
#from utils.metrics import Evaluator
from utils.meter import AverageMeter,accuracy,intersectionAndUnion
import numpy as np
from PIL import Image

#For global test
from Tester import Tester
import argparse
from tensorboardX import SummaryWriter

#For loss and scheduler
from utils.loss import CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax, FocalLoss
from utils.scheduler import Poly, OneCycle
import models

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.epochs = args.epochs
        
        self.train_data = AerialDataset(args,mode='train')
        self.train_loader =  DataLoader(self.train_data,batch_size=args.train_batch_size,shuffle=True,
                          num_workers=2)
        if args.model == 'deeplabv3+':
            self.model = models.DeepLab(num_classes=args.num_of_class,backbone='resnet')
        elif args.model == 'gcn':
            self.model = models.GCN(num_classes=args.num_of_class)
        elif args.model == 'pspnet':
            raise NotImplementedError
        else:
            raise NotImplementedError

        if args.loss == 'CE':
            self.criterion = CrossEntropyLoss2d()
        elif args.loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'LS':
            self.criterion = LovaszSoftmax()
        elif args.loss == 'F':
            self.criterion = FocalLoss()
        elif args.loss == 'CE+D':
            self.criterion = CE_DiceLoss()
        else:
            raise NotImplementedError
        
        self.num_miniepoch = get_test_times(6000,6000,args.crop_size,args.crop_size) #Only for Potsdam
        print(f'recommended to set {self.num_miniepoch} as times for iters...')
        #e.g. if crop_size=512, mini_epoch_size should be 144
        iters_per_epoch =  self.num_miniepoch * len(self.train_loader) #dataloader has already considered batch-size

        self.schedule_mode = args.schedule_mode
        self.optimizer = opt.AdamW(self.model.parameters(),lr=args.lr)
        if self.schedule_mode == 'step':
            self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.schedule_mode == 'miou' or self.schedule_mode == 'acc':
            self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.1)
        elif self.schedule_mode == 'poly':
            self.scheduler = Poly(self.optimizer,num_epochs=args.epochs,iters_per_epoch=iters_per_epoch)
        else:
            raise NotImplementedError
        
        self.cuda = args.cuda
        if self.cuda is True:
            self.model = self.model.cuda()

        self.resume = args.resume
        if self.resume != None:
            if self.cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu') 
            self.model.load_state_dict(checkpoint['parameters'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            #start from next epoch
        else:
            self.start_epoch = 1
        self.writer = SummaryWriter(comment='-'+self.model.__class__.__name__+'_'+args.loss)
        self.init_eval = args.init_eval
        
    #Note: self.start_epoch and self.epochs are only used in run() to schedule training & validation
    def run(self):
        if self.init_eval: #init with an evaluation
            init_test_epoch = self.start_epoch - 1
            Acc,mIoU,roadIoU = self.eval_complete(init_test_epoch,True)
            self.writer.add_scalar('eval/Acc', Acc, init_test_epoch)
            self.writer.add_scalar('eval/mIoU', mIoU, init_test_epoch)
            self.writer.add_scalar('eval/roadIoU',roadIoU,init_test_epoch)
            self.writer.flush()
        end_epoch = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch,end_epoch):  
            loss = self.train(epoch)
            self.writer.add_scalar('train/lr',self.optimizer.state_dict()['param_groups'][0]['lr'],epoch)
            self.writer.add_scalar('train/loss',loss,epoch)
            self.writer.flush()
            saved_dict = {
                'model': self.model.__class__.__name__,
                'epoch': epoch,
                'parameters': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            torch.save(saved_dict, f'./{self.model.__class__.__name__}_epoch{epoch}.pth.tar')
            
            Acc,mIoU,roadIoU = self.eval_complete(epoch)
            self.writer.add_scalar('eval/Acc',Acc,epoch)
            self.writer.add_scalar('eval/mIoU',mIoU,epoch)
            self.writer.add_scalar('eval/roadIoU',roadIoU,epoch)
            self.writer.flush()
            if self.schedule_mode == 'step' or self.schedule_mode == 'poly':
                self.scheduler.step()
            elif self.schedule_mode == 'miou':
                self.scheduler.step(mIoU)
            elif self.schedule_mode == 'acc':
                self.scheduler.step(Acc)
            else:
                raise NotImplementedError
        self.writer.close()

    def train(self,epoch):
        self.model.train()
        print(f"----------epoch {epoch}----------")
        print("lr:",self.optimizer.state_dict()['param_groups'][0]['lr'])
        total_loss = 0
        miniepoch_size = len(self.train_loader)
        print("#batches:",miniepoch_size*self.num_miniepoch)
        for each_miniepoch in range(self.num_miniepoch):
            for i,[img,gt] in enumerate(self.train_loader):
                print("epoch:",epoch," batch:",miniepoch_size*each_miniepoch+i+1)
                print("img:",img.shape)
                print("gt:",gt.shape)
                self.optimizer.zero_grad()
                if self.cuda:
                    img,gt = img.cuda(),gt.cuda()
                pred = self.model(img)
                print("pred:",pred.shape)
                loss = self.criterion(pred,gt.long())
                print("loss:",loss)
                total_loss += loss.data
                loss.backward()
                self.optimizer.step()
        return total_loss
    
    def eval_complete(self,epoch,save_flag=True):
        args = argparse.Namespace()
        args.by_trainer = True
        args.crop_size = self.args.crop_size
        args.stride = self.args.crop_size//2
        args.cuda = self.args.cuda
        args.model = self.model
        args.eval_list = self.args.eval_list
        args.img_path = self.args.img_path
        args.gt_path = self.args.gt_path
        args.num_of_class = self.args.num_of_class
        tester = Tester(args)
        Acc,mIoU,roadIoU=tester.run(train_epoch=epoch,save=save_flag)
        return Acc,mIoU,roadIoU

if __name__ == "__main__":
   print("--Trainer.py--")
   
