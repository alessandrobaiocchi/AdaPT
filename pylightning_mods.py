

import torch
import pytorch_lightning as pl
from torch import optim
from model import Pct, Pct_nogroup
from data import ModelNet40
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util import cal_loss, DiceLoss
import sklearn.metrics as metrics
import torch.nn.functional as F


class Lightning_pct(pl.LightningModule):
    def __init__(self, args, nclasses=40):
        super().__init__()
        self.model = Pct(args, output_channels=args.nclasses)
        self.criterion = cal_loss
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=nclasses)


    def forward(self, batch):
        data, label = batch
        data = data.permute(0, 2, 1)
        label = label.squeeze()
        logits = self.model(data)[0]
        return logits
    
    def forward_with_tokens(self, batch):
        data, label = batch
        data = data.permute(0, 2, 1)
        label = label.squeeze()
        logits, tokens = self.model(data)
        return logits, tokens
    
    def training_step(self, batch,batch_idx):
            #data, label = data.to(device), label.to(device).squeeze() 
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
     
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    


    def configure_optimizers(self):
        if self.args.train.use_sgd:
            opt = optim.SGD(self.parameters(), lr=self.args.train.lr*100, momentum=self.args.train.momentum, weight_decay=5e-4)
        else:
            opt = optim.Adam(self.parameters(), lr=self.args.train.lr, weight_decay=1e-4) 
        scheduler = CosineAnnealingLR(opt, self.args.train.epochs, eta_min=self.args.train.lr)
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    
    def validation_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
          
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('test/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

class Lightning_pct_adaptive(pl.LightningModule):
    def __init__(self, args, nclasses=40, teacher=None):
        super().__init__()
        self.model = Pct(args, output_channels=args.nclasses)
        self.criterion = cal_loss
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=nclasses)
        self.teacher = teacher


    def forward_with_mask(self, batch, drop_temp=1.0):
        data, label = batch
        data = data.permute(0, 2, 1)
        label = label.squeeze()
        
        return self.model(data, drop_temp)
    
    def forward(self, batch, drop_temp=1.0):
        return self.forward_with_mask(batch, drop_temp)[0]

    def training_step(self, batch,batch_idx):
            #data, label = data.to(device), label.to(device).squeeze() 
        labels = batch[1].squeeze()
        if self.args.train.adaptive.drop_warmup:
            lin = (self.current_epoch-self.args.train.adaptive.drop_slow_start)/(self.args.train.adaptive.drop_slow_end-self.args.train.adaptive.drop_slow_start)
            drop_temp = min(1.0, lin) * (self.current_epoch>=self.args.train.adaptive.drop_slow_start)
        else:
            drop_temp = 1.0
        ret = self.forward_with_mask(batch, drop_temp)
        if self.teacher is not None:
            with torch.no_grad():
                logits_teacher, tokens_teacher = self.teacher_module.forward_with_mask(batch)
        logits = ret[0]
        masks = ret[1]
        tokens = ret[3]
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]


        for i, m in enumerate(masks):
            m_mean = 1-m.mean(dim=1, keepdims=True)
            loss += self.args.train.adaptive.alpha * (torch.clamp((self.args.train.adaptive.drop_ratio[i]-m_mean),min=0)**2).mean() / len(masks)
            self.log(f'train/drop{i}', m_mean.mean().detach().item(), on_step=False, on_epoch = True, prog_bar=False)

        if self.teacher is not None:
            logits = torch.sigmoid(logits)
            logits_teacher = torch.sigmoid(logits_teacher)
            l_kl = F.kl_div(torch.log(torch.cat([logits, 1-logits], dim=1)), torch.log(torch.cat([logits_teacher, 1-logits_teacher], dim=-1)), reduction='batchmean', log_target=True)

            cls_token_mse = torch.sum(torch.pow(tokens[:,0,:]-tokens_teacher[:,0,:], 2).mean(dim=1))

            tokens = tokens[:,1:,:]
            tokens_teacher = tokens_teacher[:,1:,:]

            mask = masks[-1]
            B, N, C = tokens.size()
            bool_mask = mask.reshape(B*N) > 0.5
            tokens = tokens.reshape(B*N, C)[bool_mask]
            tokens_teacher = tokens_teacher.reshape(B*N, C)[bool_mask]
            l_distill = torch.pow(tokens - tokens_teacher, 2).mean()

            loss += self.args.teacher.kl_alpha * l_kl + self.args.teacher.clstkn_alpha * cls_token_mse + self.args.teacher.distill_alpha * l_distill


        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.args.train.use_sgd:
            opt = optim.SGD(self.parameters(), lr=self.args.train.lr*100, momentum=self.args.train.momentum, weight_decay=5e-4)
        else:
            opt = optim.Adam(self.parameters(), lr=self.args.train.lr, weight_decay=1e-4) 
        scheduler = CosineAnnealingLR(opt, self.args.train.epochs, eta_min=self.args.train.lr)
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    
    def validation_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        if self.args.train.adaptive.drop_warmup:
            lin = (self.current_epoch-self.args.train.adaptive.drop_slow_start)/(self.args.train.adaptive.drop_slow_end-self.args.train.adaptive.drop_slow_start)
            drop_temp = min(1.0, lin) * (self.current_epoch>=self.args.train.adaptive.drop_slow_start)
        else:
            drop_temp = 1.0
        ret = self.forward_with_mask(batch, drop_temp)
        logits = ret[0]
        masks = ret[1]
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        for i, m in enumerate(masks):
            m_mean = 1-m.mean(dim=1, keepdims=True)
            loss += self.args.train.adaptive.alpha * (torch.clamp((self.args.train.adaptive.drop_ratio[i]-m_mean),min=0)**2).mean() / len(masks)
            self.log(f'val/drop{i}', m_mean.mean().detach().item(), on_step=False, on_epoch = True, prog_bar=False)
        
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('test/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss



class Lightning_pct_merger(pl.LightningModule):
    def __init__(self, args, nclasses=40):
        super().__init__()
        self.model = Pct(args)
        self.criterion = cal_loss
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=nclasses)


    def forward(self, batch):
        data, label = batch
        data = data.permute(0, 2, 1)
        label = label.squeeze()
        
        logits = self.model(data)
        return logits 

    def training_step(self, batch,batch_idx):
            #data, label = data.to(device), label.to(device).squeeze() 
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
    
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.args.train.use_sgd:
            opt = optim.SGD(self.parameters(), lr=self.args.train.lr*100, momentum=self.args.train.momentum, weight_decay=5e-4)
        else:
            opt = optim.Adam(self.parameters(), lr=self.args.train.lr, weight_decay=1e-4) 
        scheduler = CosineAnnealingLR(opt, self.args.train.epochs, eta_min=self.args.train.lr)
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    
    def validation_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('test/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class Lightning_pct_merger_segm(pl.LightningModule):
    def __init__(self, args, nclasses=40):
        super().__init__()
        self.model = Pct_nogroup(args)
        self.criterion = DiceLoss()
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=nclasses)


    def forward(self, batch):
        data, label = batch
        data = data.permute(0, 2, 1)
        label = label.squeeze()
        
        logits = self.model(data)
        return logits 

    def training_step(self, batch,batch_idx):
            #data, label = data.to(device), label.to(device).squeeze() 
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
    
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('train/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.args.train.use_sgd:
            opt = optim.SGD(self.parameters(), lr=self.args.train.lr*100, momentum=self.args.train.momentum, weight_decay=5e-4)
        else:
            opt = optim.Adam(self.parameters(), lr=self.args.train.lr, weight_decay=1e-4) 
        scheduler = CosineAnnealingLR(opt, self.args.train.epochs, eta_min=self.args.train.lr)
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    
    def validation_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch[1].squeeze()
        logits = self.forward(batch)
        loss = self.criterion(logits, labels)
        preds = logits.max(dim=1)[1]
        acc = self.acc(preds, labels)
        #bal_acc = metrics.balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('test/perclassacc', bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
