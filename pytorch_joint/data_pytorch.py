# -*- coding: utf-8 -*
import os
from gensim.models import KeyedVectors
from collections import Counter
import numpy as np

word2vec_path = 'Word2Vec'
w2v_path = os.path.join(word2vec_path, "w2v_word_py3_baidu_0810")
word_vectors = KeyedVectors.load(w2v_path)
vocab = word_vectors.vocab
num_dimensions = word_vectors.syn0.shape[1]

index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def load_data(in_file):
    input_seq = []
    output_seq = []
    intent = []

    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            input_seq.append([c for c in line[0].split()])
            output_seq.append([c for c in line[1].split()[:-1]])
            intent.append([line[1].split()[-1]])
    return input_seq, output_seq, intent


PAD_IDX = 0
UNK_IDX = 1


def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1  # 得到的格式是， 'collections.Counter'
    ls = word_count.most_common(max_words)  # 返回的是一个列表，列表里面元素的格式是 元组
    total_words = len(ls) + 2
    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict["<UNK>"] = UNK_IDX
    word_dict["<PAD>"] = PAD_IDX
    return word_dict, total_words


def encode(en_sentences,
           cn_sentences,
           intent,
           en_dict,
           cn_dict,
           intent_dict,
           sort_by_len=True):

    out_en_sentences = [[en_dict.get(w, 1) for w in sent]
                        for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 1) for w in sent]
                        for sent in cn_sentences]

    intent_en = [[intent_dict.get(w, 1) for w in sent] for sent in intent]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
        intent_en = [intent_en[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences, intent_en


# 生成每个batch里选哪一个句子
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  #x_mask


def gen_examples(en_sentences, cn_sentences, intent, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_intent = [intent[t] for t in minibatch]
        mb_intent = np.array(mb_intent)

        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len, mb_intent))
    return all_ex


class VocabWord2vec(object):
    def __init__(self):
        self.vocab_size = len(vocab)

        print("vocab_size: {}".format(self.vocab_size))
        self.token2id = {w: idx for idx, w in enumerate(vocab, 5)}
        self.id2token = {idx: w for w, idx in self.token2id.items()}

        self.token2embed = {w: word_vectors.word_vec(w) for w in vocab}

        self.pad_token = "<PAD>"
        self.token2id.update({self.pad_token: 0})
        self.id2token.update({self.token2id[self.pad_token]: self.pad_token})
        self.token2embed.update({self.pad_token: [0.0] * 150})
        self.vocab_size += +1

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
        # self.embed_dim = 150

        self.eos_token = "<BOS>"
        self.token2id.update({self.eos_token: 3})
        self.id2token.update({self.token2id[self.eos_token]: self.eos_token})
        self.token2embed.update({self.eos_token: [0.0] * 150})
        self.vocab_size += 1

        self.eos_token = "<EOS>"
        self.token2id.update({self.eos_token: 4})
        self.id2token.update({self.token2id[self.eos_token]: self.eos_token})
        self.token2embed.update({self.eos_token: [0.0] * 150})
        self.vocab_size += 1

        self.embeddings = [
            self.token2embed[self.id2token[i]] for i in range(self.vocab_size)
        ]
        self.PAD_ID = self.token2id[self.pad_token]

        assert len(self.token2id) == len(self.id2token) == len(
            self.token2embed) == self.vocab_size
