from __future__ import print_function, absolute_import
import os.path as osp
import re

from reid.utils.data import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification.

    Dataset statistics:
    # identities: 4101
    # images: 126441
    """
    dataset_dir = 'MSMT17_V1'

    def __init__(self, root, verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.verbose = verbose
        self.load()
    
    def load(self, verbose=True):
        train_dir = osp.join(self.dataset_dir, 'list_train.txt')
        val_dir = osp.join(self.dataset_dir, 'list_val.txt')
        self.train, train_pids, train_cams = self._pluck_msmt(train_dir, 'train')
        self.val, val_pids, val_cams = self._pluck_msmt(val_dir, 'train')
        self.train = self.train + self.val

        query_dir = osp.join(self.dataset_dir, 'list_query.txt')
        gallery_dir = osp.join(self.dataset_dir, 'list_gallery.txt')
        self.query, _, _ = self._pluck_msmt(query_dir, 'test')
        self.gallery, _, _ = self._pluck_msmt(gallery_dir, 'test')
        self.num_train_pids = len(list(set(train_pids).union(set(val_pids))))
        self.num_train_cams = len(list(set(train_cams).union(set(val_cams))))

        if self.verbose:
            print(self.__class__.__name__, "v1~~~ dataset loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _pluck_msmt(self, list_file, subdir,
                    pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        ret = []
        pids_ = []
        cams_ = []
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[0]
            pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
            if pid not in pids_:
                pids_.append(pid)
            if cam not in cams_:
                cams_.append(cam)

            ret.append((osp.join(self.dataset_dir,subdir,fname), pid, cam))

        return ret, pids_, cams_
