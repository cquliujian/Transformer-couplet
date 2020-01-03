"""
2019.12.06 by liu

"""

import tensorflow as tf
import hparams
import hparams
import model_helper

hp = hparams.HParams().hparams


class Transformer:
    def __init__(self):
        # 字典部分
        self.embedding = model_helper.get_token_embedding(hp.vocab_size, hp.embedding_size, zero_pad=True)
        self.BEG_ID = 2
        self.EOF_ID = 3
        self.UNK_ID = 1
        self.PAD = 0

    def encode(self, xs, is_train = True):
        """
        Transformer 编码部分
        :param in_id:
        :param is_train:
        :return:
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            in_id, in_length = xs
            in_id = tf.cast(in_id, tf.int32)
            in_length = tf.cast(in_length, tf.int32)
            in_vector = tf.nn.embedding_lookup(self.embedding, in_id)   # [batch_size, max_length, embedding_size]

            in_vector *= hp.d_model**0.5  # trying weight-

            in_vector += model_helper.positional_encoding(in_vector, hp.max_in_length)

            in_vector = tf.layers.dropout(in_vector, rate=hp.dropout_rate, training=is_train)

            for i in range(hp.encode_layer):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    in_vector = model_helper.multihead_attention(queryies=in_vector,
                                                                 keys=in_vector,
                                                                 values=in_vector,
                                                                 num_head=hp.num_heads,
                                                                 dropout_rate=hp.dropout_rate,
                                                                 training=is_train,
                                                                 causality=False)
                    in_vector = model_helper.feed_forward(in_vector, num_units=[hp.d_ff, hp.d_model])

        memory = in_vector
        return memory

    def decode(self, ys, menmory, is_train=True):

        max_out_length = hp.max_in_length  # 对联性质特殊，输入输出长度一致
        dec = tf.nn.embedding_lookup(self.embedding, ys)
        dec *= hp.d_model ** 0.5

        dec += model_helper.positional_encoding(dec, max_out_length)

        dec = tf.layers.dropout(dec, hp.dropout_rate, training=is_train)

        for i in range(hp.decode_layer):
            with tf.variable_scope("num_block_{}".format(i), reuse=tf.AUTO_REUSE):
                dec = model_helper.multihead_attention(queryies=dec,
                                                       keys=dec,
                                                       values=dec,
                                                       num_head=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=is_train,
                                                       causality=True,
                                                       scope="self_attention")

                dec = model_helper.multihead_attention(queryies=dec,
                                                       keys=menmory,
                                                       values=menmory,
                                                       num_head=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=is_train,
                                                       causality=False,
                                                       scope="vanilla_attention"
                                                       )
                dec = model_helper.feed_forward(dec, num_units=[hp.d_ff, hp.d_model])

        weights = tf.transpose(self.embedding)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat

    def get_loss_train_op(self, xs, ys):

        y_id, y = ys[0], ys[1]
        PAD_ID = 0
        memory = self.encode(xs)
        logit, pred = self.decode(y_id, memory)

        y_ = model_helper.label_smoothing(tf.one_hot(y, depth=hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_)
        nonpadding = tf.cast(tf.not_equal(y, PAD_ID), tf.float32)
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()

        lr = model_helper.noam_scheme(hp.lr, global_step, hp.warmup_steps)
        # lr = hp.lr
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        return loss, train_op

    def predict(self, xs):

        memory = self.encode(xs, is_train=False)
        decoder_input = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.BEG_ID  # [max_length, 1]
        old_decoder_input = decoder_input
        for _ in range(hp.max_in_length):
            logit, y_hat = self.decode(decoder_input, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.PAD:  break

            decoder_input = tf.concat((old_decoder_input, y_hat), 1)

        return y_hat