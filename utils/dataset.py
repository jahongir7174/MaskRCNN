import random

import numpy
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from utils import util


@DATASETS.register_module()
class MOSAICDataset:
    def __init__(self, dataset, image_sizes, pipeline):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.pipeline = Compose(pipeline)
        if hasattr(self.dataset, 'flag'):
            self.flag = numpy.zeros(len(dataset), dtype=numpy.uint8)
        self.image_sizes = image_sizes
        self.num_samples = len(dataset)
        self.indices = range(len(dataset))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        while True:
            if random.random() > 0.25:
                data = util.mosaic(self, index)
            else:
                data = util.mix_up(self, index, random.choice(self.indices))

            if data is None:
                index = random.choice(self.indices)
                continue

            return util.process(self, data)


def build_dataset(cfg, default_args=None):
    if cfg['type'] == 'MOSAICDataset':
        import copy
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        return MOSAICDataset(**cp_cfg)
    else:
        from mmdet.datasets import builder
        return builder.build_dataset(cfg, default_args)
