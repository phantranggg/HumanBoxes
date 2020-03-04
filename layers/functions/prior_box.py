import torch
from itertools import product as product
import numpy as np


class PriorBox(object):
    def __init__(self, cfg, box_dimension=None, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.variance = cfg['variance']
        self.densities = cfg['densities']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        if phase == 'train':
            self.image_size = (cfg['min_dim'], cfg['min_dim'])
            self.feature_maps = cfg['feature_maps']
        elif phase == 'test':
            self.feature_maps = box_dimension.cpu().numpy().astype(np.int)
            self.image_size = image_size
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            densities = self.densities[k]
            min_sizes = self.min_sizes[k]
            aspect_ratios = self.aspect_ratios[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size, aspect_ratio, density in zip(min_sizes, aspect_ratios, densities):
                    s_kx = min_size / self.image_size[1]
                    s_ky = aspect_ratio * min_size / self.image_size[0]
                    offsets = [(0.5 + d) / density for d in range(density)]

                    dense_cx = [(j + x)*self.steps[k]/self.image_size[1] for x in offsets]
                    dense_cy = [(i + y)*self.steps[k]/self.image_size[0] for y in offsets]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                        
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
