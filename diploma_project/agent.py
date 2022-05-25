# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt


def variable_summaries(name, var, with_max_min=False):
    """ TensorBoard 可视化 """

    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)

        if with_max_min:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))


class Seq2seqModel:

    def __init__(self, config, input_, input_len_, mask):

        self.action_size = config.num_cpus
        self.batch_size = config.batch_size
        self.embeddings = config.embedding_size
        self.state_size = config.num_vnfd
        self.length = config.max_length
        vocab_size = config.num_vnfd + 1  # 生成context vector长度

        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers

        self.positions = []
        self.outputs = []

        self.input_ = input_
        self.input_len_ = input_len_
        self.mask = mask

        self.initialization_stddev = 0.1
        self.attention_plot = []

        with tf.variable_scope("actor"):

            # encoder
            with tf.variable_scope("actor_encoder"):

                # 初始化
                initializer = tf.contrib.layers.xavier_initializer()

                # embedding
                embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embeddings], -1.0, 1.0),
                                         dtype=tf.float32)

                embedded_input = tf.nn.embedding_lookup(embeddings, input_)

                # 生成多层 LSTM cell
                enc_cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True) for _ in range(self.num_layers)],
                    state_is_tuple=True)

                c_initial_states = []
                h_initial_states = []

                # 初始化LSTM状态c&h
                for i in range(self.num_layers):
                    first_state = tf.get_variable("var{}".format(i), [1, self.hidden_size], initializer=initializer)
                    # first_state = tf.Print(first_state, ["first_state", first_state], summarize=10)

                    c_initial_state = tf.tile(first_state, [self.batch_size, 1])
                    h_initial_state = tf.tile(first_state, [self.batch_size, 1])

                    c_initial_states.append(c_initial_state)
                    h_initial_states.append(h_initial_state)

                # 多层LSTM
                rnn_tuple_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(c_initial_states[idx], h_initial_states[idx])
                     for idx in range(self.num_layers)])

                # LSTM output
                self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=enc_cells,
                                                                                   inputs=embedded_input,
                                                                                   initial_state=rnn_tuple_state,
                                                                                   dtype=tf.float32)
            # decoder
            with tf.variable_scope("actor_decoder"):

                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True) for _ in range(self.num_layers)],
                    state_is_tuple=True)

                first_process_block_input = tf.tile(tf.Variable(tf.random_normal([1, self.hidden_size]),
                                                                name='first_process_block_input'), [self.batch_size, 1])

                # attention
                with tf.variable_scope("actor_attention_weights", reuse=True):
                    W_ref = tf.Variable(
                        tf.random_normal([self.hidden_size, self.hidden_size], stddev=self.initialization_stddev),
                        name='W_ref')
                    W_q = tf.Variable(
                        tf.random_normal([self.hidden_size, self.hidden_size], stddev=self.initialization_stddev),
                        name='W_q')
                    v = tf.Variable(tf.random_normal([self.hidden_size], stddev=self.initialization_stddev), name='v')

                decoder_state = self.encoder_final_state
                decoder_input = tf.unstack(self.encoder_outputs, num=None, axis=1,
                                           name='unstack')  # decoder中第一个时间步也要作为attention输入

                decoder_outputs = []
                decoder_attLogits = []

                # 根据SFC链长度循环产生下一个预测的序列值
                for t in range(self.length):
                    decoder_output, decoder_state = decoder_cell(inputs=decoder_input[t], state=decoder_state)

                    # dec_output = tf.layers.dense(dec_output, self.embedding, tf.nn.relu)
                    decoder_outputs.append(decoder_output)
                    # decoder_input = decoder_output
                    # _, attnLogits, context = self.attention(W_ref, W_q, v, attnInputs=enc_outputs, query=dec_output, mask=self.mask)
                    # dec_attLogits.append(attnLogits)
                    # dec_input = context

                dec_outputs = tf.stack(decoder_outputs, axis=1)
                # self.attention_plot = tf.stack(dec_attLogits, axis=1)

            # decoder的输出，大小为 batch_size x sfc_length x num_cpus,输出每个对应的概率：
            # 其中sfc_length x num_cpus就代表sfc中的每个vnf放置在不同cpu的概率
            self.decoder_logits = tf.layers.dense(dec_outputs, self.action_size)  # [Batch, seq_length, action_size]

            self.decoder_softmax = tf.nn.softmax(self.decoder_logits)
            # 根据softmax输出对防止策略采样
            prob = tf.contrib.distributions.Categorical(probs=self.decoder_softmax)

            # 产生1维的分布，输出即为placement，需要转成int
            self.decoder_exploration = prob.sample(1)
            self.decoder_exploration = tf.cast(self.decoder_exploration, tf.int32)

            # Decoder 预测，使用argmax寻找概率最大的索引，即寻找每个放置概率最大的cpu——得到placement
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
            self.decoder_prediction = tf.expand_dims(self.decoder_prediction, 0)

            # 采样
            temperature = 15
            self.decoder_softmax_temp = tf.nn.softmax(self.decoder_logits / temperature)

            prob = tf.contrib.distributions.Categorical(probs=self.decoder_softmax)

            # 从decoder预测的placement中采样16个返回
            self.samples = 16
            self.decoder_sampling = prob.sample(self.samples)

    def attention(self, W_ref, W_q, v, attnInputs, query, mask=None, maskPenalty=10 ^ 6):

        with tf.variable_scope("RNN_Attention"):
            u_i0s = tf.einsum('kl,itl->itk', W_ref, attnInputs)
            u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
            unscaledAttnLogits = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s))

            if mask is not None:
                maskedUnscaledAttnLogits = unscaledAttnLogits - tf.multiply(mask, maskPenalty)

            attnLogits = tf.nn.softmax(maskedUnscaledAttnLogits)

            context = tf.einsum('bi,bic->bc', attnLogits, attnInputs)

        return unscaledAttnLogits, attnLogits, context

    # def plot_attention(self, attention):
    #     """ Plot attention """
    #
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.matshow(attention, cmap='viridis')
    #
    #     fontdict = {'fontsize': 14}
    #
    #     ax.set_xticklabels(['sentence'], fontdict=fontdict, rotation=90)
    #     ax.set_yticklabels(['predicted'], fontdict=fontdict)
    #
    #     plt.show()


class ValueEstimator():

    def __init__(self, config, input_):
        with tf.variable_scope("value_estimator"):
            self.embeddings = config.embedding_size
            self.length = config.max_length
            vocab_size = config.num_vnfd + 1

            self.target = tf.placeholder(tf.float32, [config.batch_size], name="target")

            # Embeddings
            embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embeddings], -1.0, 1.0),
                                     dtype=tf.float32)

            embedded_input = tf.nn.embedding_lookup(embeddings, input_)

            # Encoder
            encoder_cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)

            _, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, embedded_input, dtype=tf.float32)

            output = tf.layers.dense(encoder_final_state.h, 1)

            self.value_estimate = tf.squeeze(output)
            target = self.target

            self.loss = tf.squared_difference(self.value_estimate, target)
            # variable_summaries('valueEstimator_loss', self.loss, with_max_min=False)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            self.train_op = self.optimizer.minimize(  # 没写var_list默认计算图中所有变量
                self.loss, global_step=tf.contrib.framework.get_global_step())


class Agent:

    def __init__(self, config):
        # Training config (agent)
        self.learning_rate = config.learning_rate
        # self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        # self.lr_start = config.lr1_start  # initial learning rate
        # self.lr_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        # self.lr_decay_step = config.lr1_decay_step  # learning rate decay step

        self.action_size = config.num_cpus
        self.batch_size = config.batch_size
        self.embeddings = config.embedding_size
        self.state_size = config.num_vnfd
        self.length = config.max_length

        # 计算违反限制的惩罚权重
        self.lambda_occupancy = 1000
        self.lambda_bandwidth = 10
        self.lambda_latency = 50

        self.input_ = tf.placeholder(tf.int32, [self.batch_size, self.length], name="input")
        self.input_len_ = tf.placeholder(tf.int32, [self.batch_size], name="input_len")
        self.mask = tf.placeholder(tf.int32, [self.batch_size, self.length], name="mask")  # 防止padding的区域参与注意力运算

        self._build_model(config)
        self._build_ValueEstimator(config)
        self._build_optimization()

        self.merged = tf.summary.merge_all()

    def _build_model(self, config):
        with tf.variable_scope('actor'):
            self.actor = Seq2seqModel(config, self.input_, self.input_len_, self.mask)

    def _build_ValueEstimator(self, config):
        with tf.variable_scope('value_estimator'):
            self.valueEstimator = ValueEstimator(config, self.input_)

    def _build_optimization(self):
        with tf.name_scope('reinforce_learning'):
            self.placement_holder = tf.placeholder(tf.float32, [self.batch_size, self.length], name="placement_holder")
            self.baseline_holder = tf.placeholder(tf.float32, [self.batch_size], name="baseline_holder")
            self.target_holder = tf.placeholder(tf.float32, [self.batch_size], name="target_holder")

            # 自动更新学习率（Adam）
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99, epsilon=0.0000001)

            # Multinomial distribution
            probs = tf.contrib.distributions.Categorical(probs=self.actor.decoder_softmax)
            log_softmax = probs.log_prob(self.placement_holder)  # [Batch, seq_length]

            log_softmax_mean = tf.reduce_mean(log_softmax, 1)  # [Batch]
            variable_summaries('log_softmax_mean', log_softmax_mean, with_max_min=True)

            self.advantage = self.target_holder - self.baseline_holder
            variable_summaries('adventage', self.advantage, with_max_min=False)

            # 计算loss，将target-基线后得到的值乘log_softmax的均值
            self.loss_rl = tf.reduce_mean(self.advantage * log_softmax_mean, 0)

            tf.summary.scalar('loss', self.loss_rl)

            # 根据loss目标函数计算梯度。没写var_list默认计算图中所有变量
            gvs = opt.compute_gradients(self.loss_rl)

            # 梯度裁剪，最大值1
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2
            # 使用计算得到的梯度来更新
            self.train_step = opt.apply_gradients(capped_gvs)


if __name__ == "__main__":
    pass
