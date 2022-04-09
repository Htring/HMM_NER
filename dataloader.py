#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/04/09
@description:
"""
import os


class HMMDataLoader(object):
    __doc__ = """ HMM的数据加载器 """

    def __init__(self, corpus_dir="data/corpus"):
        self.data_dir = corpus_dir
        self.token2id = {}
        self.tag2id = {}
        self.train_data, self.dev_data, self.test_data = None, None, None
        self._build_data()

    @staticmethod
    def _load_data(file_path):
        with open(file_path, 'r', encoding='utf8') as reader:
            return reader.readlines()

    @staticmethod
    def _get_data_list(data_list):
        token_list, tag_list = [], []
        tmp_token_list, tmp_tag_list = [], []
        for line in data_list:
            line = line.strip()
            if not line:
                if tmp_token_list:
                    token_list.append(tmp_token_list)
                    tag_list.append(tmp_tag_list)
                    tmp_token_list = []
                    tmp_tag_list = []
            else:
                token, tag = line.split()
                tmp_token_list.append(token)
                tmp_tag_list.append(tag)
        if tmp_token_list:
            token_list.append(tmp_token_list)
            tag_list.append(tmp_tag_list)
        return token_list, tag_list


    def _build_data(self):
        train_data_list = self._load_data(os.path.join(self.data_dir, "train.char.bmes"))
        dev_data_list = self._load_data(os.path.join(self.data_dir, "dev.char.bmes"))
        test_data_list = self._load_data(os.path.join(self.data_dir, "test.char.bmes"))
        # build vocab
        self.train_data = self._get_data_list(train_data_list)
        self.dev_data = self._get_data_list(dev_data_list)
        self.test_data = self._get_data_list(test_data_list)
        token_set, tag_set = set(), set()
        for data in [self.train_data, self.dev_data, self.test_data]:
        # for data in [self.train_data,]:
            for tokens, tags in zip(*data):
                token_set.update(tokens)
                tag_set.update(tags)

        token_set = list(token_set)
        token_set.sort()
        tag_set = list(tag_set)
        tag_set.sort()
        self.token2id = {token: index for index, token in enumerate(token_set)}
        self.tag2id = {tag: index for index, tag in enumerate(tag_set)}
