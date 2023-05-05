import json
import operator
import numpy as np
import os
import itertools
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from det3d.datasets.base import BaseDataset

class NuImageDataset(BaseDataset):

    def __init__(self,
                info_path,
                root_path,
                nsweeps,
                loading_pipelines = None,
                augmentation = None,
                prepare_label = None,
                class_names=[],
                evaluations=None,
                version="v1.0-trainval"):
        
        super(NuImageDataset, self).__init__(
            root_path, info_path, loading_pipelines, augmentation, prepare_label, evaluations)

        self.version = version

   