# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import sys
import time
import random
import argparse
import numpy as np
import collections
import os.path as osp
from datetime import timedelta
from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F

from reid import models
from reid import datasets
from reid.utils.logging import Logger
from reid.models.cm import ClusterMemory
from reid.trainers import ContrastTrainer
from reid.models.prompter import Prompter
from reid.evaluators import Evaluator, extract_features
from reid.utils.faiss_rerank import compute_jaccard_distance
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.train_utils import get_test_loader, get_train_loader, reservior_sampling,\
                                   get_mean_std_groups

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset


def separate_train(dataset, stages):
    trainset = dataset.train
    num_train_pids = dataset.num_train_pids
    num_sep_pids = num_train_pids // stages
    dataset.trains = []
    for sep_idx in range(stages):
        sep_train = []
        for item in trainset:
            if item[1] >= num_sep_pids * sep_idx and\
                item[1] < num_sep_pids * (sep_idx + 1):
                sep_train.append(item)
        dataset.trains.append(sep_train)
    return dataset


def create_model(args):
    model = models.create(
        args.arch, num_features=args.features, norm=True, 
        dropout=args.dropout, num_classes=0
    )
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


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
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    test_loader_source = get_test_loader(
        dataset_source, args.height, args.width, args.batch_size, args.workers
    )
    dataset_target = get_data(args.dataset_target, args.data_dir)
    dataset_target = separate_train(dataset_target, args.stages)
    test_loader = get_test_loader(
        dataset_target, args.height, args.width, args.batch_size, args.workers
    )

    mean_groups = []
    std_groups = []
    if args.trans and args.cas:
        mean_groups, std_groups = get_mean_std_groups(args, dataset_source)
    mean_std_groups = [mean_groups, std_groups]

    # Create model
    model = create_model(args)
    if args.prompter:
        prompter = Prompter().cuda()
        prompter.load_state_dict(torch.load(args.prompter_path))
    else:
        prompter = None

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        source_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  source mAP {:.1%}"
              .format(start_epoch, source_mAP))

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1
    )

    # Trainer
    trainer = ContrastTrainer(model, trans=args.trans,
                              mean_std_groups=mean_std_groups)

    if args.replay:
        replay_memory, seen_ids = reservior_sampling([], dataset_source.train, seen_ids=0)

    for stage in range(args.stages):
        print('======= Begining stage {} ======='.format(stage))
        for epoch in range(args.epochs):
            with torch.no_grad():
                print('==> Create pseudo labels for unlabeled data')
                cluster_loader = get_test_loader(
                    dataset_target, args.height, args.width,
                    args.batch_size, args.workers, testset=sorted(dataset_target.trains[stage])
                )

                features, _ = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat(
                    [features[f].unsqueeze(0) for f, _, _ in sorted(dataset_target.trains[stage])], 0
                )
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

                if epoch == 0:
                    # DBSCAN cluster
                    eps = args.eps
                    print('Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist)
                num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            if args.replay:
                pseudo_labeled_dataset = [] + replay_memory
                label_offset = max([item[1] for item in replay_memory]) + 1 if len(replay_memory) > 0 else 0
            else:
                pseudo_labeled_dataset = []
                label_offset = 0
            for ((fname, _, cid), label) in zip(sorted(dataset_target.trains[stage]), pseudo_labels):
                if label != -1:
                    pseudo_labeled_dataset.append((fname, label.item() + label_offset, cid))

            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            if args.replay and len(replay_memory) > 0:
                cluster_memory_loader = get_test_loader(
                    dataset_target, args.height, args.width,
                    args.batch_size, args.workers, testset=sorted(replay_memory)
                )
                memory_features, memory_labels = extract_features(
                    model, cluster_memory_loader, print_freq=50
                )
                memory_features = torch.cat(
                    [memory_features[f].unsqueeze(0) for f, _, _ in sorted(replay_memory)], 0
                )
                memory_labels = torch.cat(
                    [memory_labels[f].unsqueeze(0) for f, _, _ in sorted(replay_memory)], 0
                ).numpy()
                features = torch.cat((features, memory_features), dim=0)
                pseudo_labels = np.concatenate((pseudo_labels + label_offset, memory_labels))
                print('==> Replay merged: {} clusters'.format(num_cluster + label_offset))

            cluster_features = generate_cluster_features(pseudo_labels, features)
            del cluster_loader, features

            # Create hybrid memory
            memory = ClusterMemory(
                model.module.num_features, num_cluster+label_offset, temp=args.temp,
                momentum=args.momentum, use_hard=args.use_hard
            ).cuda()
            memory.features = F.normalize(cluster_features, dim=1).cuda()
            trainer.memory = memory

            train_loader = get_train_loader(
                dataset_target, args.height, args.width, args.batch_size, args.workers,
                args.num_instances, iters, trainset=pseudo_labeled_dataset
            )

            train_loader.new_epoch()

            trainer.train(epoch, train_loader, optimizer, prompter=prompter,
                          print_freq=args.print_freq, train_iters=len(train_loader))

            if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
                mAP = evaluator.evaluate(
                    test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=False
                )
                evaluator.evaluate(
                    test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=False
                )
                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}\n'.
                      format(epoch, mAP, best_mAP))

            lr_scheduler.step()
        
        if args.replay:
            replay_memory, seen_ids = reservior_sampling(
                replay_memory, dataset_target.trains[stage], seen_ids=seen_ids
            )

        # save the last checkpoint for each stage
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_mAP': best_mAP,
        }, is_best=False, fpath=osp.join(args.logs_dir, 'checkpoint_{}.pth.tar'.format(stage)))

    print('=== Performance Conclusion ===')
    for stage in range(args.stages):
        print('========== Stage {} ==========='.format(stage))
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint_{}.pth.tar'.format(stage)))
        model.load_state_dict(checkpoint['state_dict'])
        print('======= Target dataset =======')
        evaluator.evaluate(test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=True)
        print('======= Source dataset =======')
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--replay', action='store_true', default=False)
    parser.add_argument('--trans', action='store_true', default=False)
    parser.add_argument('--prompter', action='store_true', default=False)
    parser.add_argument('--cas', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default="", metavar='PATH')
    parser.add_argument('--prompter-path', type=str, default="", metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../../data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    main()
