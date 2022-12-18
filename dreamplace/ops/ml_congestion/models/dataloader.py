import os.path as osp
import os
import copy
import numpy as np
from torchvision.transforms import Compose

class CongestionDataset(object):
    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, ispd_dir=None, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        self.ispd_dir = ispd_dir
        self.data_infos = self.load_annotations() + self.load_ispd_benchmark()
        print(len(self.data_infos))

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                    data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos
    
    def load_ispd_benchmark(self):
        data_infos = []
        if self.ispd_dir is not None:
            for file in os.listdir(os.path.join(self.ispd_dir, 'feature')):
                feature_path = os.path.join(self.ispd_dir, 'feature', file)
                label_path = os.path.join(self.ispd_dir, 'label', file)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        print(len(data_infos))
        return data_infos


    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])

        results = self.pipeline(results) if self.pipeline else results
        
        feature =  results['feature'].transpose(2, 0, 1).astype(np.float32)
        label = results['label'].transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)