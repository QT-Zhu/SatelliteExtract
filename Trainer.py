from torch.utils.data.dataloader import DataLoader
from utils.AerialDataset import AerialDataset
import torch
import os
import torch.nn as nn
import torch.optim as opt
from utils.utils import ret2mask
from utils.meter import AverageMeter,accuracy,intersectionAndUnion
import numpy as np
from PIL import Image


from tensorboardX import SummaryWriter

#For loss and scheduler
from utils.loss import CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax, FocalLoss
from utils.scheduler import Poly
import models

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.epochs = args.epochs
        
        self.train_data = AerialDataset(args,mode='train')
        self.train_loader =  DataLoader(self.train_data,batch_size=args.train_batch_size,shuffle=True,
                          num_workers=2)
        self.eval_data = AerialDataset(args,mode='eval')
        self.eval_loader =  DataLoader(self.eval_data,batch_size=args.eval_batch_size,shuffle=False,
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

        self.token = self.model.__class__.__name__+'_'+args.loss

        self.optimizer = opt.AdamW(self.model.parameters(),lr=args.lr)
        self.scheduler = Poly(self.optimizer,num_epochs=args.epochs,iters_per_epoch=len(self.train_loader))
        
        self.cuda = args.cuda
        if self.cuda is True:
            self.model = nn.DataParallel(self.model)
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
        self.writer = SummaryWriter(comment='-'+self.token)
        self.init_eval = args.init_eval
        
    #Note: self.start_epoch and self.epochs are only used in run() to schedule training & validation
    def run(self):
        if self.init_eval: #init with an evaluation
            init_test_epoch = self.start_epoch - 1
            Acc,mIoU,roadIoU = self.eval(init_test_epoch)
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
                'model': self.token,
                'epoch': epoch,
                'parameters': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            torch.save(saved_dict, f'./{self.token}_epoch{epoch}.pth.tar')

            Acc, mIoU, roadIoU = self.eval(epoch)
            self.writer.add_scalar('eval/Acc',Acc,epoch)
            self.writer.add_scalar('eval/mIoU',mIoU,epoch)
            self.writer.add_scalar('eval/roadIoU',roadIoU,epoch)
            self.writer.flush()
            self.scheduler.step()
            
        self.writer.close()

    def train(self,epoch):
        self.model.train()
        print(f"----------epoch {epoch}----------")
        print("lr:",self.optimizer.state_dict()['param_groups'][0]['lr'])
        total_loss = 0
        for i,[img,gt] in enumerate(self.train_loader):
            print("epoch:",epoch," batch:",i+1)
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
    
    def eval(self,epoch):
        self.model.eval()
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        print(f"-------eval epoch {epoch}--------")
        for i,[img,gt] in enumerate(self.eval_loader):
            print("epoch:",epoch," batch:",i+1)
            print("img:",img.shape)
            print("gt:",gt.shape)
            if self.cuda:
                img,gt = img.cuda(),gt.cuda()
            pred = self.model(img)
            print("pred:",pred.shape)

            ret = torch.argmax(pred,dim=1).data.detach().cpu().numpy()
            gt = gt.data.detach().cpu().numpy()

            acc, pix = accuracy(ret,gt)
            intersection, union = intersectionAndUnion(ret,gt,self.args.num_of_class)
            acc_meter.update(acc,pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        roadIoU = 0
        for i, _iou in enumerate(iou):
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
            if i==1:
                roadIoU = _iou
        mIoU = iou.mean()
        Acc = acc_meter.average()
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}'.format(mIoU,Acc))
        print('Road IoU: {:.4f}'.format(roadIoU))
        return Acc,mIoU,roadIoU


if __name__ == "__main__":
   print("--Trainer.py--")
   
