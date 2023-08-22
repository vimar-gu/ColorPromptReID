import torch
import numpy as np
import collections
from torch.utils.data import DataLoader

import reid.utils.data.transforms as T

from reid.utils.data import IterLoader
from reid.utils.color_transformer import ColorTransformer
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor, TwinsPreprocessor


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    basic_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(TwinsPreprocessor(train_set, root=dataset.images_dir, transform=train_transformer,
                                     basic_transform=basic_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_raw_loader(dataset, height, width, batch_size, workers, testset):
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def sample_selection(data, num_samples):
    data = np.array(data)
    if len(data) >= num_samples:
        return data[np.random.choice(len(data), num_samples)].tolist()
    else:
        return data[np.random.choice(len(data), num_samples, replace=True)].tolist()


def reservior_sampling(memory_data, new_data, memory_size=512, num_instances=8, seen_ids=0):
    '''ID-wise Reservior Sampling
    memory_data: list
    new_data: list
    '''
    num_pids = memory_size // num_instances
    pid_dict = collections.defaultdict(list)
    for item in new_data:
        pid_dict[item[1]].append(item)
    pid_set = list(pid_dict.keys())
    if len(memory_data) == 0:
        pid_cnt = 0
    else:
        pid_cnt = max([item[1] for item in memory_data]) + 1
    for pid in pid_set:
        # memory not full, just append
        if len(memory_data) < memory_size:
            sel_data = sample_selection(pid_dict[pid], num_instances)
            sel_data = [(item[0], pid_cnt, int(item[2])) for item in sel_data]
            memory_data += sel_data
        # memory is full, random replacing
        else:
            random_num = np.random.choice(seen_ids)
            if random_num < num_pids:
                sel_data = sample_selection(pid_dict[pid], num_instances)
                # avoid id overlap
                sel_data = [(item[0], pid_cnt, int(item[2])) for item in sel_data]
                memory_data[random_num * num_instances: (random_num + 1) * num_instances] = sel_data
        seen_ids += 1
        pid_cnt += 1

    # relabel to restrict the identity
    new_pid_set = list(set([item[1] for item in memory_data]))
    relabel_dict = {pid: idx for idx, pid in enumerate(new_pid_set)}
    memory_data = [(item[0], relabel_dict[item[1]], item[2]) for item in memory_data]

    print('==> Finish Memory Construction, Size {}'.format(len(memory_data)))

    return memory_data, seen_ids


def get_mean_std_groups(args, dataset_source):
    mean_groups = []
    std_groups = []
    color_transformer = ColorTransformer()
    cluster_loader_source = get_raw_loader(
        dataset_source, args.height, args.width, args.batch_size,
        args.workers, testset=dataset_source.train
    )
    img_dict = collections.defaultdict(list)
    for item in cluster_loader_source:
        img_src = item[0]
        camids_src = item[-2]
        for camid, img in zip(camids_src, img_src):
            img_dict[camid.item()].append(img)
    for camid in img_dict.keys():
        cam_imgs = torch.stack(img_dict[camid]).cuda()
        means, _, stds, _ = color_transformer.get_mean_std_dist(
            cam_imgs, lab=True)
        mean_groups.append(means)
        std_groups.append(stds)
    mean_groups = torch.stack(mean_groups).cuda()
    std_groups = torch.stack(std_groups).cuda()

    return mean_groups, std_groups
