## 数据来源
本程序数据来源于：[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition).
为了能够使用seqeval工具评估模型效果，将原始数据中“M-”开头的标签处理为“I-”.

程序设计介绍可参考我的博文：[【NLP】基于隐马尔可夫模型（HMM）的命名实体识别（NER）实现](https://piqiandong.blog.csdn.net/article/details/124065834)

## 模型训练和模型使用
在trainer.py文件中实现了基于HMM的模型训练，以及模型使用：
```python
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
    # hmm_train()
    mode_use()

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