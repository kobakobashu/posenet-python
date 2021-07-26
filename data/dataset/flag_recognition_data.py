# -*- coding: utf-8 -*-
"""FlagRecognitionData dataset"""

import os
import logging
import json
import random
import math

import torch
import numpy as np
from torchvision import transforms
import requests

import csv
import pprint


log = logging.getLogger(__name__)


class FlagRecognitionData(torch.utils.data.Dataset):
    """FlagRecognitionData dataset"""

    def __init__(self, cfg: object, mode: str) -> None:
        """Initialization
    
        Get dataset.
        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.
        """
        with open('/workspace/datasets/flag_recognition_data/all.csv') as f:
            reader = csv.reader(f)

        #json_open = open('/workspace/datasets/flag_recognition_data/data.json', 'r')
        #roadmaps = json.load(json_open)

            x = []
            y = []
            for row in reader:
                row = list(map(float, row))
                x.append([row[i * 51 : (i+1) * 51] for i in range(10)])
                y.append(row[-1])
            log.info(x[0])
        """
        len_input = cfg.data.dataset.len_input

        for roadmap in roadmaps:
            roadmap = [0]*len_input + roadmap

            for i in range(len_input, len(roadmap)):
                x.append(roadmap[i-len_input:i])
                y.append(roadmap[i])
        """

        self.x = torch.tensor(x)
        self.y = torch.tensor(y).long()
        self.data_len = len(self.x)
        log.info(self.x.shape)
        log.info(self.y.shape)
    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        out_data = self.x[idx]
        out_label =  self.y[idx]

        return out_data, out_label