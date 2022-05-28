#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2022/04/09
@description:
"""
import torch
from hmm import HMM
from dataloader import HMMDataLoader
from seqeval.metrics import classification_report
import json


def evaluate(model: HMM, token_lists, tags_lists, id2tag):
    """
    模型评估
    :param id2tag:
    :param model:
    :param token_lists: token
    :param tags_lists:  real tag
    :return:
    """
    predict_list = []
    real_tag_list = []
    for index, tokens in enumerate(token_lists):
        pred_tag_list = []
        for id_ in model.forward(torch.tensor(tokens)):
            pred_tag_list.append(id2tag.get(id_.item()))
        predict_list.append(pred_tag_list)
        real_tags = [id2tag.get(id_) for id_ in tags_lists[index]]
        real_tag_list.append(real_tags)

    print("\n", classification_report(y_true=real_tag_list, y_pred=predict_list))
    return predict_list


def model_save(model: HMM,
               token2id,
               id2tag,
               model_path="ckpt/hmm.pt",
               token2id_path="ckpt/token2id.json",
               id2tag_path="ckpt/id2tag.json"):
    """
    模型存储
    :param model: Hmm 模型
    :param id2tag: id转tag字典
    :param token2id: token2id 字典
    :param token2id_path: token2id字典保存地址
    :param id2tag_path: id2tag的字典保存地址
    :param model_path:模型存储地址
    :return:
    """

    with open(id2tag_path, 'w', encoding='utf8') as writer:
        json.dump(id2tag, writer, ensure_ascii=False)

    with open(token2id_path, 'w', encoding='utf8') as writer:
        json.dump(token2id, writer, ensure_ascii=False)

    model = torch.jit.script(model)
    model.save(model_path)


def hmm_train():
    hmm_dataloader = HMMDataLoader()
    hmm = HMM(len(hmm_dataloader.token2id), len(hmm_dataloader.tag2id))
    hmm.fit(*hmm_dataloader.train_data)
    evaluate(hmm, *hmm_dataloader.test_data, hmm_dataloader.id2tag)
    model_save(hmm,
               hmm_dataloader.token2id,
               hmm_dataloader.id2tag)


def pt_model_use():

    def _load_model():
        with open("ckpt/token2id.json", 'r', encoding='utf8') as reader:
            t2i = json.load(reader)
        with open("ckpt/id2tag.json", 'r', encoding='utf8') as reader:
            i2t = json.load(reader)
            i2t = {int(id_): value for id_, value in i2t.items()}

        return torch.jit.load("ckpt/hmm.pt"), t2i, i2t

    hmm, token2id, id2tag = _load_model()
    test_data = "常建良，男，"
    num_data = [token2id.get(char, -1) for char in  list(test_data)]
    res = hmm.forward(torch.tensor(num_data))
    decode_res = []
    for item in res:
        decode_res.append(id2tag.get(item.item()))
    print(decode_res)


if __name__ == '__main__':
    # hmm_train()
    pt_model_use()
