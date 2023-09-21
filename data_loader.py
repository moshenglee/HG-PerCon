#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : lml
# @File    : data_loader.py
# @Software: PyCharm
"""
数据导入，准备各层输入所需数据等，包括
user_word_adj、word_pern_adj
pern_pern_adj、 label、pern_feature_256
"""
import os

import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset

from utils import mylogger

logger = mylogger()


class InteractionDataSet(Dataset):
    def __init__(self, file_dir):
        #  "input/youtube_gcn_1"
        input_dir = file_dir[:5]
        self.user_word_adj = np.load(os.path.join(file_dir, "user_word_adj.npy"))
        logger.info("user_word_adj loaded! " + str(self.user_word_adj.shape))

        self.word_pern_adj = np.load(os.path.join(input_dir, "word_pern_adj.npy"))
        logger.info("word_pern_adj loaded! " + str(self.word_pern_adj.shape))

        self.pern_adj = np.load(os.path.join(input_dir, "pern_pern_adj.npy"))
        logger.info("pern_pern_adj loaded! " + str(self.pern_adj.shape))

        # build symmetric adjacency matrix
        # self.word_pern_adj += np.multiply(self.word_pern_adj.T, self.word_pern_adj.T > self.word_pern_adj) - \
        #                       np.multiply(self.word_pern_adj, self.word_pern_adj.T > self.word_pern_adj)
        # self.pern_adj += np.multiply(self.pern_adj.T, self.pern_adj.T > self.pern_adj) - \
        # np.multiply(self.pern_adj, self.pern_adj.T > self.pern_adj)
        # self-loop trick, the input graphs should have no self-loop
        self.word_pern_adj += np.identity(self.word_pern_adj.shape[1])
        self.word_pern_adj[self.word_pern_adj != 0] = 1.0

        self.pern_adj += np.identity(self.pern_adj.shape[1])
        self.pern_adj[self.pern_adj != 0] = 1.0

        self.word_pern_adj = self.word_pern_adj.astype(np.dtype('B'))
        self.pern_adj = self.pern_adj.astype(np.dtype('B'))

        logger.info("graphs laplacian loaded!")
        # label_list = ['ope', 'con', 'ext', 'agr', 'neu']
        # idx = 0
        self.labels = np.load(os.path.join(file_dir,"label.npy")).astype(np.float64)
        # self.labels = self.labels.reshape(-1, 1)
        logger.info("labels loaded! " + str(self.labels.shape))

        pern_features = np.load(os.path.join(input_dir, "pern_feature_256.npy"))
        self.pern_features = preprocessing.scale(pern_features)
        logger.info("global personality features loaded! " + str(self.pern_features.shape))
        word_features = np.random.rand(89, self.pern_features.shape[-1])
        self.word_features = preprocessing.scale(word_features)
        logger.info("global random word features loaded! " + str(self.word_features.shape))
        self.N = self.user_word_adj.shape[0]

    def get_pern_features(self):
        return self.pern_features

    def get_word_features(self):
        return self.word_features

    def get_feature_dimension(self):
        return self.pern_features.shape[-1]

    def get_word_pern_adj(self):
        return self.word_pern_adj

    def get_pern_adj(self):
        return self.pern_adj

    def get_user_word_adj(self):
        return self.user_word_adj

    def get_labels(self):
        return self.labels

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.user_word_adj[idx], self.labels[idx]
