from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.utils.logging import Logger
from reid.evaluators import Evaluator
from reid.utils.data import IterLoader
from reid.models.prompter import Prompter
from reid.utils.data import transforms as T
from reid.trainers import PreTrainer, PrompterTrainer
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor, TwinsPreprocessor
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR


best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers,
             num_instances, iters=200):
    root = osp.join(data_dir)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.create(name, root)

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.596, 0.558, 0.497])
         ])
    basic_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor()
    ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(dataset.train, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(
            TwinsPreprocessor(
                dataset.train, root=dataset.images_dir, transform=train_transformer,
                basic_transform=basic_transformer
            ), batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=not rmgs_flag, pin_memory=True, drop_last=True
        ), length=iters
    )

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, dataset.num_train_pids, train_loader, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_source, num_classes, train_loader_source, test_loader_source = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances,
                 args.iters)

    dataset_target, _, _, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, args.iters)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)
    model.cuda()
    model = nn.DataParallel(model)

    if args.prompter:
        prompter = Prompter().cuda()
        prompter_trainer = PrompterTrainer(prompter)
    else:
        prompter = None

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test on source domain:")
        evaluator.evaluate(
            test_loader_source, dataset_source.query, dataset_source.gallery,
            cmc_flag=True, rerank=args.rerank, data=args.dataset_source
        )
        print("Test on target domain:")
        evaluator.evaluate(
            test_loader_target, dataset_target.query, dataset_target.gallery,
            cmc_flag=True, rerank=args.rerank, data=args.dataset_target
        )
        return

    params = []
    for _, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr,
                    "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(
        optimizer, args.milestones, gamma=0.3, warmup_factor=0.01,
        warmup_iters=args.warmup_step
    )

    # Trainer
    trainer = PreTrainer(model, num_classes, margin=args.margin, trans=args.trans)

    # Start training
    for epoch in range(0, args.epochs):
        train_loader_source.new_epoch()

        trainer.train(
            epoch, train_loader_source, optimizer,
            train_iters=len(train_loader_source), print_freq=args.print_freq
        )
        if args.prompter:
            prompter_trainer.train(epoch, train_loader_source,
                                   train_iters=len(train_loader_source))

        lr_scheduler.step()

        if (epoch + 1) % args.eval_step == 0:
            _, mAP = evaluator.evaluate(test_loader_source, dataset_source.query,
                                        dataset_source.gallery, cmc_flag=True,
                                        data=args.dataset_source)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            if args.prompter:
                torch.save(prompter_trainer.prompter.state_dict(),
                           osp.join(args.logs_dir, 'prompter.pth.tar'))

            print('\n * Finished epoch {:3d} source mAP: {:5.1%} best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print("Test on target domain:")
    evaluator.evaluate(
        test_loader_target, dataset_target.query, dataset_target.gallery,
        cmc_flag=True, rerank=args.rerank, data=args.dataset_target
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--trans', action='store_true', default=False)
    parser.add_argument('--prompter', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default="", metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--margin', type=float, default=0.0,
                        help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='../data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='../logs')

    main()
