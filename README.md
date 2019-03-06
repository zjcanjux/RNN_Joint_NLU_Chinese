# Joint_NLU Chinese data 中文语料训练

## Discription

Encoder使用`tf.nn.bidirectional_dynamic_rnn`实现

Decoder使用`tf.contrib.seq2seq.dynamic_decode`实现

## Usage

Put Chinese Word embeddings file into Word2Vec folder

```
python main.py
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

Reference:

- [Tensorflow动态seq2seq使用总结（r1.3）](https://github.com/applenob/RNN-for-Joint-NLU/blob/master/tensorflow_dynamic_seq2seq.md)
- https://github.com/HadoopIt/rnn-nlu)

