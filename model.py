# coding=utf-8
# @author: cer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell,DropoutWrapper,MultiRNNCell
import sys


class Model:
    def __init__(self, sess,input_steps, embedding_size,embeddings, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size):
        self.sess = sess
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.embedding = embeddings

        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        # 每句输入的实际长度，除了padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='decoder_targets')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')
        self.global_step = tf.Variable(0, trainable = False)
        print('vocab_size : {}'.format(self.vocab_size))

    def build(self):
        # get embeddings and convert encoder inputs to encoder inputs embeddings
        # self.embeddings = tf.get_variable("embeddings",shape=[self.vocab_size, self.embedding_size],
        #                                     initializer=tf.constant_initializer(self.embedding),trainable=False)
        self.embeddings = tf.get_variable("embeddings",shape=[self.vocab_size, self.embedding_size],
                                            initializer=tf.constant_initializer(np.array(self.embedding)),trainable=False)
        
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        # Encoder
        # 使用单个LSTM cell
        encoder_f_cell_0 = LSTMCell(self.hidden_size)
        encoder_b_cell_0 = LSTMCell(self.hidden_size)
        encoder_f_cell = DropoutWrapper(encoder_f_cell_0,output_keep_prob=0.5)
        encoder_b_cell = DropoutWrapper(encoder_b_cell_0,output_keep_prob=0.5)

        # 下面四个变量的尺寸：T*B*D，T*B*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        


        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)


        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        # Decoder
        decoder_lengths = self.encoder_inputs_actual_length

        intent_W_1 = tf.Variable(tf.random_uniform([self.hidden_size * 2, 100], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W_1")
        intent_b_1 = tf.Variable(tf.zeros([100]), dtype=tf.float32, name="intent_b_1")

        intent_W_2 = tf.Variable(tf.random_uniform([100, self.intent_size], -0.1, 0.1), dtype =tf.float32, name ='intent_W_2')
        intent_b_2 = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name = 'intent_b_2')

        intent_hidden_1 = tf.nn.relu(tf.add(tf.matmul(encoder_final_state_h, intent_W_1), intent_b_1))
        intent_logits = tf.add(tf.matmul(intent_hidden_1, intent_W_2), intent_b_2)

        # intent_prob = tf.nn.softmax(intent_logits)
        self.intent = tf.argmax(intent_logits, axis=1)


        sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='SOS') * 2
        sos_step_embedded = tf.nn.embedding_lookup(self.embeddings, sos_time_slice)

        # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        pad_step_embedded = tf.zeros([self.batch_size, self.hidden_size*2+self.embedding_size],
                                     dtype=tf.float32)

        def initial_fn():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # 选择logit最大的下标作为sample
            print("outputs", outputs)
            # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
            # print("slot output_logits: ", output_logits)
            # prediction_id = tf.argmax(output_logits, axis=1)
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
            # 输入是h_i+o_{i-1}+c_i
            next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state

        my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                memory = tf.transpose(encoder_outputs, [1, 0, 2])

                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size, memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)

                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_size)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.slot_size, reuse=reuse
                )

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size))

                # initial_state=encoder_final_state)
                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=True,
                    impute_finished=True, maximum_iterations=self.input_steps
                )
                return final_outputs


        outputs = decode(my_helper, 'decode')

        print("outputs: ", outputs)
        print("outputs.rnn_output: ", outputs.rnn_output)
        print("outputs.sample_id: ", outputs.sample_id)
        # weights = tf.to_float(tf.not_equal(outputs[:, :-1], 0))
        self.decoder_prediction = outputs.sample_id
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
        self.decoder_targets_time_majored = tf.transpose(self.decoder_targets, [1, 0])
        self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
        print("decoder_targets_true_length: ", self.decoder_targets_true_length)
        # 定义mask，使padding不计入loss计算
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
        # 定义slot标注的损失
        loss_slot = tf.contrib.seq2seq.sequence_loss(
            outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)
      
        # 定义intent分类的损失
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
            logits=intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)

        self.loss = loss_slot + 1.01*loss_intent

        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        
        # self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars), global_step = self.global_step)

        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.vars])
        print('There are {} parameters in the model'.format(param_num))


        # self.train_op = optimizer.minimize(self.loss)
        # train_op = layers.optimize_loss(
        #     loss, tf.train.get_global_step(),
        #     optimizer=optimizer,
        #     learning_rate=0.001,
        #     summaries=['loss', 'learning_rate'])


    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        
        if mode == 'train':

            output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                self.intent, self.mask]

            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1],
                         self.decoder_targets: unziped[2], 
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.decoder_prediction, self.intent]

            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results
