#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2022/04/09
@description:
"""
import pickle
from hmm import HMM
from dataloader import HMMDataLoader


def hmm_train():
    hmm_dataloader = HMMDataLoader()
    hmm = HMM(hmm_dataloader.token2id, hmm_dataloader.tag2id)
    hmm.fit(*hmm_dataloader.train_data)
    hmm.evaluate(*hmm_dataloader.test_data)
    hmm.model_save("ckpt/hmm.pkl")

def mode_use():
    with open("ckpt/hmm.pkl", 'rb') as reader:
        hmm: HMM = pickle.load(reader)
    test_data = "常建良，男，"
    print(hmm.infer(list(test_data)))


if __name__ == '__main__':
    hmm_train()
    # mode_use()
