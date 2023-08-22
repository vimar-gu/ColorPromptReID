import time
import torch
import numpy as np
from torchvision.utils import save_image, make_grid

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth, SoftTripletLoss
from reid.utils.meters import AverageMeter
from reid.utils.data import transforms as T
from reid.utils.color_transformer import ColorTransformer


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0, trans=False):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tr = SoftTripletLoss(margin=margin).cuda()
        self.trans = trans
        self.color_transformer = ColorTransformer()
        self.train_transformer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    def train(self, epoch, loader_source, optimizer, train_iters=200,
              print_freq=100):
        self.model.train()

        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        for it in range(train_iters):
            s_inputs_o, s_inputs, targets = self._parse_data(
                loader_source.next()
            )
            s_features, s_cls_out = self.model(s_inputs, training=True)

            if self.trans:
                s_inputs_t = self.train_transformer(
                    self.color_transformer.color_transfer_shuffle(s_inputs_o)
                )
                s_features_t, s_cls_out_t = self.model(s_inputs_t, training=True)
                s_features = torch.cat((s_features, s_features_t))
                s_cls_out = torch.cat((s_cls_out, s_cls_out_t))
                targets = torch.cat((targets, targets))

            loss_ce, loss_tr, prec = self._forward(
                s_features, s_cls_out, targets
            )
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (it + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss_ce {:.3f} ({:.3f})\t'
                       'Loss_tr {:.3f} ({:.3f})\tPrec {:.2%} ({:.2%})\t'.format(
                           epoch, it + 1, train_iters,
                           losses_ce.val, losses_ce.avg,
                           losses_tr.val, losses_tr.avg,
                           precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs_o, imgs, _, pids, _, _ = inputs
        imgs_o = imgs_o.cuda()
        imgs = imgs.cuda()
        pids = pids.cuda()
        return imgs_o, imgs, pids

    def _forward(self, features, outputs, targets):
        loss_ce = self.criterion_ce(outputs, targets)
        loss_tr = self.criterion_tr(features, features, targets)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class PrompterTrainer(object):
    def __init__(self, prompter=None):
        super(PrompterTrainer, self).__init__()
        self.color_transformer = ColorTransformer()
        self.train_transformer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.prompter = prompter
    
    def train(self, epoch, loader_source, train_iters=200,
              print_freq=100):
        losses_cond = AverageMeter()

        for it in range(train_iters):
            s_inputs_o, _, _ = self._parse_data(
                loader_source.next()
            )

            trans_inputs = self.color_transformer.color_transfer_resample(s_inputs_o)
            trans_mean, trans_std = self.color_transformer.get_mean_std(
                trans_inputs, lab=True
            )
            original_mean, original_std = self.color_transformer.get_mean_std(
                s_inputs_o, lab=True
            )

            mean_gamma, mean_bias, std_gamma, std_bias = self.prompter(
                self.train_transformer(trans_inputs)
            )
            mean = trans_mean * mean_gamma + mean_bias
            std = trans_std * std_gamma + std_bias
            loss_c = torch.sum((mean - original_mean) ** 2, dim=1).mean() +\
                torch.sum((std - original_std) ** 2, dim=1).mean()
            losses_cond.update(loss_c.item())

            self.prompter.optim.zero_grad()
            loss_c.backward()
            self.prompter.optim.step()

            if (it + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss_cond {:.3f} ({:.3f})'.format(
                      epoch, it + 1, train_iters,
                      losses_cond.val, losses_cond.avg))

    def _parse_data(self, inputs):
        imgs_o, imgs, _, pids, _, _ = inputs
        imgs_o = imgs_o.cuda()
        imgs = imgs.cuda()
        pids = pids.cuda()
        return imgs_o, imgs, pids


class ContrastTrainer(object):
    def __init__(self, model, memory=None, trans=False, mean_std_groups=None):
        super(ContrastTrainer, self).__init__()
        self.model = model
        self.memory = memory
        self.trans = trans
        self.mean_groups = mean_std_groups[0]
        self.std_groups = mean_std_groups[1]
        if self.trans:
            self.color_transformer = ColorTransformer()
            self.train_transformer = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    
    def train(self, epoch, data_loader, optimizer, prompter=None,
              train_iters=400, print_freq=100):
        self.model.train()
        losses = AverageMeter()
        for it in range(train_iters):
            inputs_o, inputs, targets = self._parse_data(
                data_loader.next()
            )

            f_out = self.model(inputs)
            loss = self.memory(f_out, targets)

            if self.trans:
                mean, std = self.color_transformer.get_mean_std(
                    inputs_o, lab=True
                )
                with torch.no_grad():
                    mean_gamma, mean_bias, std_gamma, std_bias = prompter(
                        self.train_transformer(inputs_o)
                    )
                mean = mean * mean_gamma + mean_bias
                std = std * std_gamma + std_bias

                inputs_r = self.color_transformer.color_transfer_target(
                    inputs_o, mean=mean, std=std, frame=True)
                inputs_r = self.train_transformer(inputs_r)
                f_out_r = self.model(inputs_r)
                loss = loss * 0.5 + self.memory(f_out_r, targets) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if (it + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, it + 1, len(data_loader),
                              losses.val, losses.avg))
    
    def _parse_data(self, inputs):
        imgs_o, imgs, _, pids, _, _ = inputs
        imgs_o = imgs_o.cuda()
        imgs = imgs.cuda()
        pids = pids.cuda()
        return imgs_o, imgs, pids
