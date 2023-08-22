from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils.faiss_rerank import compute_jaccard_distance
from .loss.triplet import normed_cosine_dist as cosine_dist


def extract_cnn_feature(model, inputs):
    inputs = inputs.cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, choice_c=0, print_freq=100, metric=None):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, item in enumerate(data_loader):
            imgs, fnames, pids =item[0], item[1], item[choice_c + 2]
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            if isinstance(outputs, tuple):
                for fname, g_output, pid in zip(fnames, outputs[0], pids):
                    features[fname] = g_output
                    labels[fname] = pid
            else:
                for fname, output, pid in zip(fnames, outputs, pids):
                    features[fname] = output
                    labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[item[0]].unsqueeze(0) for item in query], 0)
    y = torch.cat([features[item[0]].unsqueeze(0) for item in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, data='None'):
    if query is not None and gallery is not None:
        query_ids = [item[1] for item in query]
        gallery_ids = [item[1] for item in gallery]
        query_cams = [item[-1] for item in query]
        gallery_cams = [item[-1] for item in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, data=data)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, data=data, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k,
                      cmc_scores['market1501'][k-1]))
    if (not cmc_flag):
        return mAP

    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None, data='none'):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features

        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        if (not rerank):
            results = evaluate_all(distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, data=data)

            return results

        print('Applying person re-ranking ...')
        distmat_qq,_,_ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg,_,_ = pairwise_distance(features, gallery, gallery, metric=metric)

        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())

        return evaluate_all(distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
