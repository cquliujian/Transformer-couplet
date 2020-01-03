"""
2019.12 by liu

"""

import tensorflow as tf
import hparams
import numpy as np
import data_loader

hp = hparams.HParams().hparams

def get_token_embedding(vocab_size, embeding_size, zero_pad=True):
    """
    建立word_embedding [vocab_size, embedding_size]
    :param vocab_size:
    :param embeding_size:
    :param zero_pad: if True 即矩阵第一行的值为0, PAD_ID = 0 则最后得到的词向量的值pad值对应的值为0, 方便后序对mask的处理
    :return:
    """
    with tf.variable_scope("word_embedding"):
        embedding = tf.get_variable(name="embedding",
                                    dtype=tf.float32,
                                    shape=[vocab_size, embeding_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embedding = tf.concat([tf.zeros(shape=[1, embeding_size]),
                                   embedding[1:, :]], axis=0)
    return embedding


def positional_encoding(input_vector, max_in_length, mask=True):
    """
    位置编码部分
    :param input_vector:
    :param max_in_length:
    :return:
    """
    embedding_size = hp.embedding_size
    batch_size = get_batch_size()
    max_length = max_in_length
    in_length = tf.shape(input_vector)[1]
    with tf.variable_scope("positional_encoding", reuse=tf.AUTO_REUSE):

        position_id = tf.tile(tf.expand_dims(tf.range(in_length), 0), [batch_size, 1])
        position_embedding = [[pos/np.power(10000, 2*i/embedding_size) for i in range(embedding_size)]
                              for pos in range(max_length)]  # [embedding_size, max_length]
        position_embedding = np.array(position_embedding)
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # 2i 从0开始 step=2
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # 2i+1 从1开始 step=2
        position_embedding = tf.convert_to_tensor(position_embedding)

        position_output = tf.nn.embedding_lookup(position_embedding, position_id)

        if mask:
            position_output = tf.where(tf.equal(input_vector, 0), input_vector, tf.cast(position_output, tf.float32))

    return tf.cast(position_output, tf.float32)

def multihead_attention(queryies, keys, values,
                        num_head=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queryies.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 根据论文 先接线性层
        Q = tf.layers.dense(queryies, d_model, use_bias=False)  # batch_size*max_length*d_model
        K = tf.layers.dense(keys, d_model, use_bias=False)
        V = tf.layers.dense(values, d_model, use_bias=False)

        # 多头拆分
        Q_ = tf.split(Q, num_head, axis=-1)  # 进行拆分, 得到长度为num_heade 的列表
        K_ = tf.split(K, num_head, axis=-1)
        V_ = tf.split(V, num_head, axis=-1)

        Q_ = tf.concat(Q_, axis=0)  # [batch_size*num_head, max_length, d_model/head_num]
        K_ = tf.concat(K_, axis=0)
        V_ = tf.concat(V_, axis=0)

        # 点积 attention = softmax(Q* transpose(K)/d**0.5)V
        outputs = scaled_dot_attention(Q_, K_, V_,
                                       causality=causality,
                                       dropout_rate=dropout_rate,
                                       training=training)
        outputs = tf.split(outputs, num_head, axis=0)
        outputs = tf.concat(outputs, axis=-1)

        # res
        outputs = outputs + queryies  # 残差这步加的是queries

        # 层正则
        outputs = ln(outputs)

        return outputs


def feed_forward(inputs, num_units, scope="feed_forward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 输入层
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        #输出层
        outputs = tf.layers.dense(outputs, num_units[1])

        # res
        outputs += inputs

        outputs = ln(outputs)  # normalization

    return outputs

def scaled_dot_attention(Q, K, V,
                         causality=False, dropout_rate=0.,
                         training=True,
                         scope="scaled_dot_product_attention"
                         ):
    # 点积 attention = softmax(Q* transpose(K)/d**0.5)V
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # batch_size* max_seq_length_Q* max_seq_length_K
        outputs /= d_k**0.5

        # key mask
        outputs = mask(outputs, Q, K, type="key")

        if causality:
            outputs = mask(outputs, type="feature")

        outputs = tf.nn.softmax(outputs)   # batch_size, q_length, k_length
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))  # ??? 保存attention 图片的意义
        outputs = mask(outputs, Q, K, type="query")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)   # batch_size, q_length, d_v

    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    """
    mask 的主目的是为了使pad值对应的attention值为一个非常小的数值,防止对后序logit的计算产生影响,在sql任务中用过类似的用法
    :param inputs:
    :param queries:
    :param keys:
    :param type:
    :return:
    """
    pad_num = -2**32+1  # 32位整形
    if type == "key":
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(inputs)*pad_num

        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # batch_size*length_q*length*k
    elif type == "query":
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])   # batch_size*length_q*length_k
        """
        paddings = tf.ones_like(inputs)*pad_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        """

        outputs = inputs * masks
    elif type == "feature":
        # 即当前的query 不能看到之后的值
        diag_vals = tf.ones_like(inputs[0, :, :])  # q_length, k_length
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()

        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(inputs)*pad_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        raise ValueError("scaled dot error!!! please check it")

    return outputs


def ln(inputs, epsilon=1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_shape = inputs.get_shape()
        params_shape = input_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)   # batch_size * q_length* 1
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs-mean)/((variance+epsilon)**(0.5))

        outputs = gamma * normalized + beta

    return outputs


def label_smoothing(inputs, epsilon=0.1):

    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def get_configure():
    """
    estimator config
    :return:
    """
    run_config = tf.estimator.RunConfig(model_dir=hp.model_dir,
                                        save_checkpoints_secs=None,
                                        save_checkpoints_steps=hp.save_steps,
                                        keep_checkpoint_max=hp.max_save,
                                        )
    return run_config


def id2sentence(ids_list, vocdict):
    # 将ids转化为句子

    sentence = ""
    for i in ids_list:
        if i == 0 or i == 3:
            return sentence
        sentence += vocdict[i]
    sentence+="\n"
    return sentence

def get_batch_size():
    if hp.model_mode == "train":
        return hp.batch_size
    else:
        return hp.batch_size
if __name__ == "__main__":
    vocdict = data_loader.DataLoader().voacb_list
    a = [[5, 192, 344, 23, 343, 4324, 432, 0]]
    print(id2sentence(a, vocdict))


