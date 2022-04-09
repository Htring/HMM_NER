#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: hmm.py
@time:2022/04/09
@description: 隐马尔可夫模型
"""
import pickle
import torch
from seqeval.metrics import classification_report


class HMM(object):
    __doc__ = """ 隐马尔可夫模型 """

    def __init__(self, token2id: dict, tag2id: dict):
        """
        :param token2id: 语料token到id的字典
        :param tag2id: 语料tag到id的字典
        """
        self.M = len(token2id)  # HMM模型观测数目，即语料中的字符数目
        self.N = len(tag2id)  # HMM模型状态数目，即标注的种类
        self.tag2id = tag2id
        self.token2id = token2id
        # HMM模型中的状态转移矩阵 A[i][j]表示从状态i转移到状态j的概率
        self.A = torch.zeros(self.N, self.N)
        # 观测矩阵/发射矩阵，B[i][j]表示状态i生成观测j的概率
        self.B = torch.zeros(self.N, self.M)
        # 初始状态概率，Pi[i] 表示初始时刻为状态i的概率
        self.Pi = torch.zeros(self.N)

    def fit(self, token_lists: list, tag_lists: list):
        """
        HMM模型训练，即根据语料数据对模型进行参数估计
        :param token_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
        :param tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
        :return:
        """
        assert len(token_lists) == len(tag_lists), "token len must equal to tag"
        # 状态转移矩阵计算,A
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tag_id = self.tag2id.get(tag_list[i])
                next_tag_id = self.tag2id.get(tag_list[i + 1])
                self.A[current_tag_id][next_tag_id] += 1

        # 如果某元素没有出现过，该位置为0，我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 观测概率矩阵计算, B
        for tag_list, token_list in zip(tag_lists, token_lists):
            assert len(tag_list) == len(token_list)
            for tag, token in zip(tag_list, token_list):
                tag_id = self.tag2id[tag]
                token_id = self.token2id[token]
                self.B[tag_id][token_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 初始状态矩阵计算，pi
        for tag_list in tag_lists:
            init_tag_id = self.tag2id.get(tag_list[0])
            self.Pi[init_tag_id] += 1

        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def infer(self, token_lists):
        """
        HMM模型推理
        :param token_lists: 待推理内容
        :return:
        """
        return self.decoding(token_lists)

    def evaluate(self, token_lists, tags_lists):
        """
        模型评估
        :param token_lists: token
        :param tags_list:  real tag
        :return:
        """
        predict_list = []
        for index, tokens in enumerate(token_lists):
            pred_tag_list = self.decoding(tokens)
            predict_list.append(pred_tag_list)
        print("\n", classification_report(y_true=tags_lists, y_pred=predict_list))
        return predict_list

    def decoding(self, token_list):
        """
        使用维特比算法计算给定观测序列的状态序列， 这里就是对字组成的观测序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        工程实现时考虑比较小的概率在计算时导致溢出情况，考虑使用对数的方式计算，将相乘转成相加。
        :param token_list: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
        :return:
        """
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)
        seq_len = len(token_list)
        # 初始化 维比特矩阵viterbi 维度为[状态数, 序列长度],
        # viterbi[i, j]表示标注序列的第j个标注为i的所有单序列(i_1, i_2, ..i_j)出现的概率最大值
        viterbi = torch.zeros(self.N, seq_len)
        # back_pointer是跟viterbi一样大小的矩阵
        # back_pointer[i, j]存储的是标注序列的第j个标注为i时，第j-1个标注的id
        back_pointer = torch.zeros(self.N, seq_len, dtype=torch.int)
        start_token_id = self.token2id.get(token_list[0], None)
        # 便于计算将输出矩阵转置，此时第i列即为标记字典中标记id对应输出概率，每行就是各个状态输出到某个字的概率
        Bt = B.t()
        # 字符不在字典中时，使用均值代替
        avg_score = torch.log(torch.ones(self.N) / self.N)
        # 各状态到当前字符的概率
        bt = avg_score if start_token_id is None else Bt[start_token_id]
        viterbi[:, 0] = Pi + bt  # 本质上是各状态在首位出现概率与其往首字符转移概率之积
        back_pointer[:, 0] = -1  # 首列使用-1填充
        # 遍历序列计算相关概率值
        for step in range(1, seq_len):
            token_id = self.token2id.get(token_list[step], None)
            bt = avg_score if token_id is None else Bt[token_id]
            # 遍历所有的tag，找出已有的概率与下一个状态概率的最大值
            for tag_id in range(len(self.tag2id)):
                # A[:, tag_id] 表示所有状态转移当前状态id的概率
                max_prob, max_id = torch.max(viterbi[:, step - 1] + A[:, tag_id], dim=0)
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                back_pointer[tag_id, step] = max_id

        best_path_prob, best_path_pointer = torch.max(viterbi[:, seq_len - 1], dim=0)
        # back trace
        best_path = [best_path_pointer.item()]
        for back_step in range(seq_len - 1, 0, -1):
            best_path_pointer = back_pointer[best_path_pointer, back_step]
            best_path.append(best_path_pointer.item())
        assert len(best_path) == len(token_list)
        id2tag = dict((id_, tag) for tag, id_ in self.tag2id.items())
        return [id2tag.get(id_) for id_ in reversed(best_path)]

    def model_save(self, path):
        """
        模型存储
        :param path: 模型存储地址
        :return:
        """
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)
