## 改版说明
本版本在之前的基础上支持对HMM模型转torch.script模式，调整了数据加载以及模型训练的方式。

## 数据来源
本程序数据来源于：[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition).
为了能够使用seqeval工具评估模型效果，将原始数据中“M-”开头的标签处理为“I-”.

程序设计介绍可参考我的博文：[【NLP】基于隐马尔可夫模型（HMM）的命名实体识别（NER）实现](https://piqiandong.blog.csdn.net/article/details/124065834)

## 模型训练和模型使用
在trainer.py文件中实现了基于HMM的模型训练，以及模型使用：
```python
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

```

## 模型效果
使用train.char.bmes文件内进行训练，使用test.char.bmes文件中的数据进行测试，效果如下：
```text
precision    recall  f1-score   support

        CONT       0.93      1.00      0.97        28
         EDU       0.88      0.96      0.91       112
         LOC       0.29      0.33      0.31         6
        NAME       0.81      0.81      0.81       112
         ORG       0.75      0.79      0.77       553
         PRO       0.52      0.73      0.61        33
        RACE       0.76      0.93      0.84        14
       TITLE       0.87      0.88      0.88       772

   micro avg       0.81      0.85      0.83      1630
   macro avg       0.73      0.80      0.76      1630
weighted avg       0.82      0.85      0.83      1630
```
整体效果比较可观。需要说明的是，[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition)中使用的评估方式与本程序不同，不具有可比性。
本程序以整个实体为单位进行评估。

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)