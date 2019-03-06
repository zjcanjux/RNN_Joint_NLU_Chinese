# coding=utf-8
# @author: cer
import tensorflow as tf
from data import *
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import os

input_steps = 50     
embedding_size = 150
hidden_size = 100
batch_size = 16
epoch_num = 50


log_dir = 'log_dir'
word_vocab = VocabWord2vec()
# word_vocab = VocabGlove()
embeddings = word_vocab.embeddings
vocab_size = word_vocab.vocab_size

def read_process_data():
    # train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    # test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()

    train_data = open("dataset/chinese_data/processed_data.text", "r").readlines()
    test_data = open("dataset/chinese_data/processed_data_test.text", "r").readlines()

    train_data_padded = data_padding(train_data)
    test_data_padded = data_padding(test_data)

    # index2word, index2slot,index2intent, train_data_index, test_data_index, slot_size, intent_size = get_index_data_from_glove(train_data_padded, test_data_padded)
    # 
    index2word, index2slot,index2intent, train_data_index, test_data_index, slot_size, intent_size = get_index_data_from_word2vec(train_data_padded, test_data_padded)
    print('intent_size %s'%(intent_size))
    print('slot_size %s'%(slot_size))
    return index2word, index2slot,index2intent, train_data_index, test_data_index, slot_size, intent_size

def get_model():
    model = Model(sess,input_steps, embedding_size, embeddings, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size)
    model.build()
    return model

def train(sess,index2word, index2slot,index2intent, train_data_index, test_data_index,is_debug=False):

    model = get_model()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())
    # print(tf.trainable_variables())
    
    ckpt_check = tf.train.get_checkpoint_state(log_dir, latest_filename="checkpoint")
    if ckpt_check:
        ckpt = tf.train.latest_checkpoint('log_dir')
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

    for epoch in range(epoch_num):
        train_loss = 0.0
        num_batch = 0
        # divide the train data into batches
        batch_train_index_data = getBatch(batch_size, train_data_index)
        
        # train every batch
        for _, each_batch_data in enumerate(batch_train_index_data):
            _, loss, decoder_prediction, intent, mask = model.step(sess, "train", each_batch_data)
            train_loss += loss
            num_batch +=1
        train_loss /= num_batch
        print ('global_step {}'.format(model.global_step.eval(session = sess)))
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # save model
        saver = tf.train.Saver()
        saver.save(sess, 'log_dir/my_model')

        # test after training in every epoch 
        pred_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, test_data_index)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            # decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            decoder_prediction = np.transpose(decoder_prediction)
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                sen_len = batch[index][1]
                print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                print("Intent Truth          : ", index2intent[batch[index][3]])
                print("Intent Prediction     : ", index2intent[intent[index]])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            # print("slot_pred_length: ", slot_pred_length)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            # print(np.shape(true_slot), np.shape(decoder_prediction))
            # print(true_slot, decoder_prediction)
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
            # print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        pred_slots_a = np.vstack(pred_slots)
        # print("pred_slots_a: ", pred_slots_a.shape)
        true_slots_a = np.array(list(zip(*test_data_index))[2])[:pred_slots_a.shape[0]]
        # print("true_slots_a: ", true_slots_a.shape)
        
        print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))
        print("Slot accuracy for epoch {}: {}".format(epoch, np.average(slot_accs)))
        print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))



if __name__ == '__main__':
    # train(is_debug=True)
    # test_data()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    index2word, index2slot,index2intent, train_data_index, test_data_index, slot_size, intent_size = read_process_data()
    # index2word, index2slot,index2intent, train_data_index, test_data_index = read_process_data()

    train(sess,index2word, index2slot,index2intent, train_data_index, test_data_index)
