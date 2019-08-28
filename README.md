# Joint_NLU Chinese data 中文语料训练

## Discription

Encoder使用`tf.nn.bidirectional_dynamic_rnn`实现

Decoder使用`tf.contrib.seq2seq.dynamic_decode`实现

## Usage
The file of seq_slot_intent.text is the example for data which is suitable for the B_I_slot_label.py.

Put Chinese Word embeddings file into Word2Vec folder

```
python3 tain_and_test.py
```

输出：
```
Slot Prediction       :  ['B-金额', 'O', 'O', 'O']
Intent Truth          :  开通流量
Intent Prediction     :  开通流量
Intent accuracy for epoch 4: 0.8425925925925926
Slot accuracy for epoch 4: 0.9632771300764675
Slot F1 score for epoch 4: 0.9677923702313946
global_step 5040
[Epoch 5] Average train loss: 0.09832338647591689
Input Sentence        :  ['我', '想', '更改', '宽带', '密码']
Slot Truth            :  ['O', 'O', 'O', 'O', 'B-附属标签']
Slot Prediction       :  ['O', 'O', 'O', 'O', 'B-附属标签']
Intent Truth          :  修改宽带
Intent Prediction     :  修改宽带
Intent accuracy for epoch 5: 0.8564814814814815
Slot accuracy for epoch 5: 0.9589992218974985
Slot F1 score for epoch 5: 0.9642521166509878
```

## Detail

B_I_slot_label.py       process the raw data by BIO label method 实现中文语料的标注

data.py          		    convert BIO label data into index data

## Add pytorch implementation for Joint model
见 pytorch joint folder
主要是计算效率上的小优化，每个batch计算的句子长度一样，以batch中最长的句子为基准，而不是所有batch都是统一长度。

用到词向量 链接:https://pan.baidu.com/s/1RaEcYVW5n6Dz7-GBtSP_sA  密码:unzz 
也可以根据需要用其他的词向量。

## Reference:

- [Tensorflow动态seq2seq使用总结（r1.3）](https://github.com/applenob/RNN-for-Joint-NLU/blob/master/tensorflow_dynamic_seq2seq.md)
- https://github.com/HadoopIt/rnn-nlu)

