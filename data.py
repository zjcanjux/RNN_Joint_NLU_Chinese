# coding=utf-8
# @author: cer

import json
import os
import random
import numpy as np
from gensim.models import KeyedVectors


glove_path = 'glove'
word2vec_path = 'Word2Vec'


# flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

# padding raw data
def data_padding(data, length=50):
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注

    # seq_in, seq_out, intent = data
    # data 是有N行数据，每行分别由seq_in, seq_out, intent 组成
    seq_in, seq_out, intent = list(zip(*data)) 
    seq_in_padded = []
    seq_out_padded = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        seq_in_padded.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        seq_out_padded.append(temp)
        padded_data = list(zip(seq_in_padded, seq_out_padded, intent))
    return padded_data


# generate Glove class
class VocabGlove(object):
    def __init__(self):
        word_vec_list = []
        with open(os.path.join(glove_path, "glove.6B.100d.txt")) as f:
            for line in f:
                line = line.strip().split()
                word_vec_list.append((line[0], np.array(line[1:], dtype=np.float32)))
        self.vocab_size = len(word_vec_list)

        self.token2id = {w_v[0]: i for i, w_v in enumerate(word_vec_list, 4)}
        self.id2token = {i: w for w, i in self.token2id.items()}

        self.pad_token = '<PAD>'
        self.token2id.update({self.pad_token: 0})
        self.id2token.update({self.token2id[self.pad_token]: self.pad_token})
        word_vec_list.insert(0, (self.pad_token, np.zeros(100)))
        self.vocab_size += 1

        self.unk_token = '<UNK>'
        self.token2id.update({self.unk_token: 1})
        self.id2token.update({self.token2id[self.unk_token]: self.unk_token})
        word_vec_list.insert(1, (self.unk_token, np.zeros(100)))
        self.vocab_size += 1

        self.unk_token = '<SOS>'
        self.token2id.update({self.unk_token: 2})
        self.id2token.update({self.token2id[self.unk_token]: self.unk_token})
        word_vec_list.insert(2, (self.unk_token, np.zeros(100)))
        self.vocab_size += 1

        self.unk_token = '<EOS>'
        self.token2id.update({self.unk_token: 3})
        self.id2token.update({self.token2id[self.unk_token]: self.unk_token})
        word_vec_list.insert(3, (self.unk_token, np.zeros(100)))
        self.vocab_size += 1

        word_vec_dict = dict(word_vec_list)
        self.embeddings = np.array([word_vec_dict.get(self.id2token.get(idx)) for idx in range(self.vocab_size)])
        # print("vocab_size: {}".format(self.vocab_size))

# generate word2vec class
w2v_path = os.path.join(word2vec_path, "w2v_word_py3_baidu_0810")
word_vectors = KeyedVectors.load(w2v_path)
vocab = word_vectors.vocab
num_dimensions = word_vectors.syn0.shape[1]

class VocabWord2vec(object):
    def __init__(self):
        self.vocab_size = len(vocab)

        print("vocab_size: {}".format(self.vocab_size))
        self.token2id = {w: idx for idx, w in enumerate(vocab,4)}
        self.id2token = {idx: w for w, idx in self.token2id.items()}

        self.token2embed = {w: word_vectors.word_vec(w) for w in vocab}

        self.pad_token = "<PAD>"
        self.token2id.update({self.pad_token: 0})
        self.id2token.update({self.token2id[self.pad_token]: self.pad_token})
        self.token2embed.update({self.pad_token: [0.0] * 150})
        self.vocab_size += + 1

        self.unk_token = "<UNK>"
        self.token2id.update({self.unk_token: 1})
        self.id2token.update({self.token2id[self.unk_token]: self.unk_token})
        self.token2embed.update({self.unk_token: [0.0] * 150})
        self.vocab_size += 1

        self.sos_token = "<SOS>"
        self.token2id.update({self.sos_token: 2})
        self.id2token.update({self.token2id[self.sos_token]: self.sos_token})
        self.token2embed.update({self.sos_token: [0.0] * 150})
        self.vocab_size += 1
        self.embed_dim = 150

        self.eos_token = "<EOS>"
        self.token2id.update({self.eos_token: 3})
        self.id2token.update({self.token2id[self.eos_token]: self.eos_token})
        self.token2embed.update({self.eos_token: [0.0] * 150})
        self.vocab_size += 1

        self.embeddings = [self.token2embed[self.id2token[i]] for i in range(self.vocab_size)]
        self.PAD_ID = self.token2id[self.pad_token]

        assert len(self.token2id) == len(self.id2token) == len(self.token2embed) == self.vocab_size




def get_index_data_from_glove(train_data_padded, test_data_padded):
    word_vocab = VocabGlove()
    train_index_padded_data = []
    test_index_padded_data = []

    train_seq_in, train_seq_out, train_intent = list(zip(*train_data_padded))
    test_seq_in, test_seq_out, test_intent = list(zip(*test_data_padded))

    slot_tag = set(np.array(train_seq_out + test_seq_out).flatten())
    # slot_tag = set(np.array(train_seq_out).flatten())
    slot_tag = sorted(slot_tag)
    slot_size = len(slot_tag)+1
    # print('slot_tag shape')
    # print (len(slot_tag))

    intent_tag = set(np.array(train_intent + test_intent).flatten())
    # intent_tag = set(np.array(train_intent).flatten())
    intent_tag = sorted(intent_tag)
    intent_size = len(intent_tag)+1
    # print(len(intent_tag))

    index2word = word_vocab.id2token

    # 生成tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)
    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}


    # 生成intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)
    index2intent = {v: k for k, v in intent2index.items()}

    for sin, sout, intent in train_data_padded:
        # get the value of word2index[i] 

        sin_ix = list(map(lambda i: word_vocab.token2id[i] if i in word_vocab.token2id else word_vocab.token2id["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: tag2index[i] if i in tag2index else tag2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]

        train_index_padded_data.append([sin_ix, true_length, sout_ix, intent_ix])

    for sin, sout, intent in test_data_padded:
        # get the value of word2index[i] 
        sin_ix = list(map(lambda i: word_vocab.token2id[i] if i in word_vocab.token2id else word_vocab.token2id["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: tag2index[i] if i in tag2index else tag2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        
        test_index_padded_data.append([sin_ix, true_length, sout_ix, intent_ix])

    return index2word, index2tag, index2intent,train_index_padded_data, test_index_padded_data, slot_size, intent_size





def get_index_data_from_word2vec(train_data_padded, test_data_padded):
    word_vocab = VocabWord2vec()

    train_index_padded_data = []
    test_index_padded_data = []

    train_seq_in, train_seq_out, train_intent = list(zip(*train_data_padded))
    test_seq_in, test_seq_out, test_intent = list(zip(*test_data_padded))

    slot_tag = set(np.array(train_seq_out + test_seq_out).flatten())
    # slot_tag = set(np.array(train_seq_out).flatten())
    slot_tag = sorted(slot_tag)
    slot_size = len(slot_tag)+1
    # print('slot_tag shape')
    # print (len(slot_tag))

    intent_tag = set(np.array(train_intent + test_intent).flatten())
    # intent_tag = set(np.array(train_intent).flatten())
    intent_tag = sorted(intent_tag)
    intent_size = len(intent_tag)+1
    # print(len(intent_tag))

    index2word = word_vocab.id2token

    # 生成tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)
    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}


    # 生成intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)
    index2intent = {v: k for k, v in intent2index.items()}

    for sin, sout, intent in train_data_padded:
        # get the value of word2index[i] 

        sin_ix = [word_vocab.token2id.get(w, word_vocab.token2id[word_vocab.unk_token]) for w in sin ]
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: tag2index[i] if i in tag2index else tag2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]

        train_index_padded_data.append([sin_ix, true_length, sout_ix, intent_ix])

    for sin, sout, intent in test_data_padded:
        # get the value of word2index[i] 
        sin_ix = [word_vocab.token2id.get(w, word_vocab.token2id[word_vocab.unk_token]) for w in sin ]
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: tag2index[i] if i in tag2index else tag2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        
        test_index_padded_data.append([sin_ix, true_length, sout_ix, intent_ix])

    return index2word, index2tag, index2intent,train_index_padded_data, test_index_padded_data, slot_size, intent_size





    # train_seq_in, train_seq_out, train_intent = list(zip(*train_data_padded))
    # test_seq_in, test_seq_out, test_intent = list(zip(*test_data_padded))

    # train_seq_in = [[word_vocab.token2id.get(w, word_vocab.token2id[word_vocab.unk_token]) for w in x] for x in train_seq_in]
    # train_seq_out= [[word_vocab.token2id.get(w, word_vocab.token2id[word_vocab.unk_token]) for w in x] for x in train_seq_out]
    
    # train_index_data = list(zip(train_seq_in, train_seq_out))
    return index2word, index2tag, index2intent,train_index_padded_data, test_index_padded_data, slot_size, intent_size





def getBatch(batch_size, index_data):
    random.shuffle(index_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(index_data):
        batch_index_data = index_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch_index_data