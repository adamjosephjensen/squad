# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
import pdb
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from tensor2tensor.models.transformer import TransformerEncoder
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn

logging.basicConfig(level=logging.INFO)



def shape_list(x):
    """
    taken from tensor2tensor github
    
    Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
           dim = shape[i]
        ret.append(dim)
    return ret


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.
    The padding is computed for one reference tensor containing the padding mask
    and then can be applied to any other tensor of shape [dim_origin,...].
    Ex:
            input = [
                [tok1, tok2],
                [tok3, tok4],
                [0, 0],
                [0, 0],
                [tok5, tok6],
                [0, 0],
            ]
            output = [
                [tok1, tok2],
                [tok3, tok4],
                [tok5, tok6],
            ]
    """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.
        Args:
            pad_mask (tf.Tensor): Reference padding tensor of shape
                [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
                containing non-zeros positive values to indicate padding location.
        """
        self.nonpad_ids = None
        self.dim_origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            pad_mask = tf.reshape(pad_mask, [-1])    # Flatten the batch
            # nonpad_ids contains coordinates of zeros rows (as pad_mask is
            # float32, checking zero equality is done with |x| < epsilon, with
            # epsilon=1e-9 as standard, here pad_mask only contains positive values
            # so tf.abs would be redundant)
            self.nonpad_ids = tf.to_int32(tf.where(tf.cast(pad_mask,
                                                           tf.float32) < 1e-9))
            self.dim_origin = tf.shape(pad_mask)[:1]

    def remove(self, x):
        """Remove padding from the given tensor.
        Args:
            x (tf.Tensor): of shape [dim_origin,...]
        Returns:
            a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
        """
        with tf.name_scope("pad_reduce/remove"):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(
                    x,
                    indices=self.nonpad_ids,
            )
            # if not context.in_eager_mode():
                # This is a hack but for some reason, gather_nd return a tensor of
                # undefined shape, so the shape is set up manually
            x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.
        Args:
            x (tf.Tensor): of shape [dim_compressed,...]
        Returns:
            a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
            dim is restored from the original reference tensor
        """
        with tf.name_scope("pad_reduce/restore"):
            x = tf.scatter_nd(
                    indices=self.nonpad_ids,
                    updates=x,
                    shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
            )
        return x

def transformer_ffn_layer(x, nonpadding, hidden_size, dropout=0.0):
    """
    removes padding
    computes dense-relu-dense
    then puts padding back
    """
    pad_remover = PadRemover(nonpadding)
    original_shape = shape_list(x)
    # Collapse `x` across examples, and remove padding positions.
    x = tf.reshape(x, tf.concat([[-1], original_shape[2:]],
                    axis=0))
    x = tf.expand_dims(pad_remover.remove(x), axis=0)
    x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu, name='ffn_1')
    if dropout != 0.0:
        out = tf.nn.dropout(x, dropout)
    x = tf.layers.dense(x, hidden_size, activation=None, name='ffn_2')
    out = tf.reshape(
                  pad_remover.restore(tf.squeeze(x, axis=0)),
        original_shape)
    return out



def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """ from
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    length = shape_list(x)[1]
    channels = shape_list(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale,
                                      max_timescale)
    return x + signal

class QATransformerModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the transformer-based QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                   mode='FAN_IN',
                                                                   uniform=False)
        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=init):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_transformer_b()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # 1 epoch = 640 iters with batch size 100
        start = 0.000001
        base = FLAGS.learning_rate
        warmup_steps = 3000
        slope = (base - start) / warmup_steps
        pre = slope * tf.cast(self.global_step, tf.float32) + start
        learning_rate = tf.where(
            tf.less(tf.cast(self.global_step, tf.int32), warmup_steps),
            pre, base)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    #### your code here ####
        
    def dotprod_attn(self, q, k, v, bias, name=None):
        """
        A(Q, K, V) = softmax(Q K.T / sqrt(d_k)) V

        q: a Tensor with shape [batch, heads, length_q, depth_k]
        q_padding: [batch, length_q]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        k_padding: [batch, length_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        
        returns: [batch, heads, length_q, depth_v]
        """
        with tf.variable_scope(name,
                               default_name="dot_product_attention",
                               values=[q,k,v]) as scope:
            d_k = tf.shape(k)[-1]
            print('dot product attention with \nq: {}\nk: {}, \nv: {}'.format(q, k, v))
            # QKT is [batch, heads, length_q, length_kv]
            QKT_logits = tf.matmul(q, k, transpose_b=True)
            QKT_logits = QKT_logits / tf.sqrt(tf.cast(d_k, tf.float32)) # rank 6
            print('logits:', QKT_logits)
            print('bias:', bias)
            if bias is not None:
                QKT_logits += bias # broadcasting black magic
            
            attn_dist = tf.nn.softmax(QKT_logits,
                                      name='attention_distribution')
            # returns: [batch, heads, length_q, depth_v]
            # NOTE google's implementation in tensor2tensor applies dropout to these weights
            attn_dist = tf.nn.dropout(attn_dist, self.keep_prob)
            print('attn_dist:', attn_dist)
            out = tf.matmul(attn_dist, v)
            print('weighted sum of v, based on similarity between q and k\nout:', out)
            return out


    def prepare_encoder(self, inputs, hidden_size, nonpadding):
        inputs = add_timing_signal(inputs)
        #inputs = self.compute(inputs, hidden_size, 'resize_depth')
        padding = 1 - nonpadding
        minus_large = tf.constant(-1e9, dtype=tf.float32)
        padding = tf.multiply(minus_large, tf.cast(padding, tf.float32))
        # a `Tensor` with shape [batch, 1, 1, hidden_size]
        ignore = tf.expand_dims(tf.expand_dims(padding, axis=1), axis=1) 
        encoder_self_attention_bias = ignore
        return (inputs, encoder_self_attention_bias)


    def split_last_dim(self, x, n):
        #print('split last dim in', x, 'n', n)
        x_shape = shape_list(x)
        m = x_shape[-1]
        if isinstance(m, int) and isinstance(n, int):
            if m % n != 0:
                print('m:', m, 'n:', n)
                assert( m % n == 0 )
        out = tf.reshape(x, x_shape[:-1] + [n, m // n])
        #print('split last dim out:', out)
        return out

    
    def split_heads(self, x, n_heads=None):
        if n_heads == None:
            n_heads = self.FLAGS.num_heads
        # split the last dimension and transpose
        split = self.split_last_dim(x, n_heads)
        out = tf.transpose(split, [0, 2, 1, 3])
        return out


    def combine_heads(self, x):
        # assumes x is rank 4
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        a, b = x_shape[-2:]
        out = tf.reshape(x, x_shape[0:2] + [a * b])
        # print('out in combine_hds', out)
        return out

    
    def compute(self, inputs, output_depth, name):
        """
        Output tensor the same shape as inputs except the last dimension is
        output_depth.
        """
        return tf.layers.dense(inputs,
                               output_depth,
                               use_bias=False,
                               activation=None,
                               name=name)


    def multihead_attention(self,
                            query_antecedent,
                            bias,
                            memory_antecedent,
                            total_key_depth,
                            total_value_depth,
                            output_depth):

        q = self.compute(query_antecedent, total_key_depth, 'learn_q')
        k = self.compute(memory_antecedent, total_key_depth, 'learn_k')
        v = self.compute(memory_antecedent, total_value_depth, 'learn_v')

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        x = self.dotprod_attn(q, k, v, bias)

        x = self.combine_heads(x)
        x = self.compute(x, output_depth, 'MHA_linear_output')
        return x


    def transformer_encoder_block(self,
                            query_antecedent,
                            logits_bias,
                            memory_antecedent,
                            query_mask,
                            memory_mask,
                            total_key_depth,
                            total_value_depth,
                            output_depth):
        """
        # returns [batch, query_antecedent_length, output_depth]
        """
        attn = self.multihead_attention(query_antecedent,
                                        logits_bias,
                                        memory_antecedent,
                                        total_key_depth,
                                        total_value_depth,
                                        output_depth)
        # dropout, add, and norm
        # query_antecedent might not have come in with depth == output_depth
        # but if so we add a skip connection
        attn = tf.nn.dropout(attn, self.keep_prob)
        if tf.shape(query_antecedent)[2] == tf.shape(attn)[2]:
            nor1 = tf.contrib.layers.layer_norm(attn + query_antecedent,
                                            activation_fn=tf.nn.relu)
        else:
            nor1 = tf.contrib.layers.layer_norm(attn,
                                                activation_fn=tf.nn.relu)

        # feed forward twice, without changing shape
        with tf.variable_scope("ffn"):
            fc = transformer_ffn_layer(nor1, query_mask, output_depth,
                                       self.keep_prob)
        # dropout, add, and norm
        sub2 = tf.nn.dropout(fc, self.keep_prob)
        nor2 = tf.contrib.layers.layer_norm(sub2 + nor1,
                                            activation_fn=tf.nn.relu)
        # NOTE: adding a second skip connection here is not standard but it
        # might be fun
        # if tf.shape(query_antecedent)[2] == tf.shape(nor2)[2]:
            # nor2 = tf.contrib.layers.layer_norm(nor2 + query_antecedent,
            #                                     activation_fn=tf.nn.relu)
        return nor2 



    def transformer_enc_block(self,
                              query_antecedent,
                              logits_bias,
                              query_mask,
                              total_key_depth):
        return self.transformer_encoder_block(query_antecedent,
                                      logits_bias,
                                      query_antecedent,
                                      query_mask,
                                      query_mask,
                                      total_key_depth,
                                      total_key_depth,
                                      total_key_depth)

    
    def context_to_query_attn(self, c,
                              c_mask,
                              c_bias,
                              q,
                              qn_mask,
                              qn_bias,
                              hidden_size,
                              mode='simple'):
        if mode == 'simple':
            attn_layer = BasicAttn(self.keep_prob,
                                   self.FLAGS.hidden_size,
                                   self.FLAGS.hidden_size)
            _, attn_out = attn_layer.build_graph(q, qn_mask, c)
            return attn_out
        elif mode == 'multihead':
            for i in range(self.FLAGS.num_hidden_layers):
                with tf.variable_scope("block_{}".format(i)) as scope:
                    c = self.transformer_encoder_block(c,
                                                       qn_bias,
                                                       q,
                                                       c_mask,
                                                       qn_mask,
                                                       hidden_size,
                                                       hidden_size,
                                                       hidden_size)
            return c
        else:
             print('must provide a recognized mode')
             assert(0)

    def transformer_base_v1(self):
        """Set of hyperparameters."""
        hparams = common_hparams.basic_params1()
        hparams.norm_type = "layer"
        hparams.hidden_size = 512
        hparams.batch_size = 4096
        hparams.max_length = 256
        hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
        hparams.optimizer_adam_epsilon = 1e-9
        hparams.learning_rate_schedule = "legacy"
        hparams.learning_rate_decay_scheme = "noam"
        hparams.learning_rate = 0.1
        hparams.learning_rate_warmup_steps = 4000
        hparams.initializer_gain = 1.0
        hparams.num_hidden_layers = 6
        hparams.initializer = "uniform_unit_scaling"
        hparams.weight_decay = 0.0
        hparams.optimizer_adam_beta1 = 0.9
        hparams.optimizer_adam_beta2 = 0.98
        hparams.num_sampled_classes = 0
        hparams.label_smoothing = 0.1
        hparams.shared_embedding_and_softmax_weights = True
        hparams.symbol_modality_num_shards = 16
       
        # Add new ones like this.
        hparams.add_hparam("filter_size", 2048)
        # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
        hparams.add_hparam("num_encoder_layers", 0)
        hparams.add_hparam("num_decoder_layers", 0)
        # Attention-related flags.
        hparams.add_hparam("num_heads", 8)
        hparams.add_hparam("attention_key_channels", 0)
        hparams.add_hparam("attention_value_channels", 0)
        hparams.add_hparam("ffn_layer", "dense_relu_dense")
        hparams.add_hparam("parameter_attention_key_channels", 0)
        hparams.add_hparam("parameter_attention_value_channels", 0)
        # All hyperparameters ending in "dropout" are automatically set to 0.0
        # when not in training mode.
        hparams.add_hparam("attention_dropout", 0.0)
        hparams.add_hparam("attention_dropout_broadcast_dims", "")
        hparams.add_hparam("relu_dropout", 0.0)
        hparams.add_hparam("relu_dropout_broadcast_dims", "")
        hparams.add_hparam("pos", "timing")  # timing, none
        hparams.add_hparam("nbr_decoder_problems", 1)
        hparams.add_hparam("proximity_bias", False)
        hparams.add_hparam("use_pad_remover", True)
        hparams.add_hparam("self_attention_type", "dot_product")
        hparams.add_hparam("max_relative_position", 0)
        return hparams

    def transformer_base_v2(self):
        """Set of hyperparameters."""
        hparams = self.transformer_base_v1()
        hparams.layer_preprocess_sequence = "n"
        hparams.layer_postprocess_sequence = "da"
        hparams.layer_prepostprocess_dropout = 0.1
        hparams.attention_dropout = 0.1
        hparams.relu_dropout = 0.1
        hparams.learning_rate_warmup_steps = 8000 #potentially worisome
        hparams.learning_rate = 0.2
        return hparams 


    def build_transformer_b(self):
        hidden_size = self.FLAGS.hidden_size
        hparams = self.transformer_base_v2()
        F = self.FLAGS
        hparams.batch_size = F.batch_size
        hparams.hidden_size = F.hidden_size
        hparams.num_hidden_layers = F.num_hidden_layers
        hparams.layer_prepostprocess_dropout = F.dropout
        hparams.attention_dropout = F.dropout
        hparams.relu_dropout = F.dropout
        hparams.learning_rate = F.learning_rate
        hparams.num_heads = F.num_heads
        #transformer = TransformerEncoder(hparams)
        """
        # this I tried but it had 3 million parameters?
        with tf.variable_scope("context_encoder"):
            context_embs = tf.expand_dims(self.context_embs, 2)
            context_hiddens, _ = transformer.encode(context_embs, 1, hparams)
        with tf.variable_scope("question_encoder"):
            qn_embs = tf.expand_dims(self.qn_embs, 2)
            question_hiddens, _ = transformer.encode(qn_embs, 2, hparams)
        """
        with tf.variable_scope("context_encoder"):
            # TODO replace with tensor2tensor
            context_hiddens, context_bias = self.prepare_encoder(self.context_embs,
                                                          hidden_size,
                                                          self.context_mask)
            for i in range(self.FLAGS.num_hidden_layers):
                with tf.variable_scope("block_{}".format(i)) as scope:
                    context_hiddens = self.transformer_enc_block(context_hiddens,
                                                   context_bias,
                                                   self.context_mask,
                                                   hidden_size)

        with tf.variable_scope("question_encoder"):
            question_hiddens, qn_bias = self.prepare_encoder(self.qn_embs, hidden_size,
                                                    self.qn_mask)
            for i in range(self.FLAGS.num_hidden_layers):
                with tf.variable_scope("block_{}".format(i)) as scope:
                    question_hiddens = self.transformer_enc_block(question_hiddens,
                                                   qn_bias,
                                                   self.qn_mask,
                                                   hidden_size)
        #"""

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size, self.FLAGS.hidden_size)
        _, attn_output = attn_layer.build_graph(question_hiddens,
                                                self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)

        # attention where queries come from previous layer and
        # keys and values come from the encoder output
        # weighted sum of V, based on similarity between Q and K associated
        # query-to-context attention only provides a slight benefit
        """
        SOME STUFF I TRIED THAT DID NOT WORK WELL:
        with tf.variable_scope("context_to_question_attention"):
            a = self.context_to_query_attn(c,
                                           context_mask,
                                           context_bias,
                                           q,
                                           qn_mask,
                                           qn_bias,
                                           hidden_size,
                                           'simple')
       # (batch_size, context_len, hidden_size * 3
        model = tf.concat([c, a, c * a], axis=2) 
        with tf.variable_scope("model_encoder"):
            with tf.variable_scope("block_1") as scope:
                _1 = self.transformer_enc_block(model,
                                                   context_bias,
                                                   context_mask,
                                                   hidden_size)
            with tf.variable_scope("block_2") as scope:
                _2 = self.transformer_enc_block(_1,
                                                   context_bias,
                                                   context_mask,
                                                   hidden_size)
            with tf.variable_scope("block_3") as scope:
                _3 = self.transformer_enc_block(_2,
                                                   context_bias,
                                                   context_mask,
                                                   hidden_size)
                                                       
        with vs.variable_scope("StartDist"):
            _1_2 = tf.concat([_1, _2], axis=2)
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(_1_2, self.context_mask)

        with vs.variable_scope("EndDist"):
            _1_3 = tf.concat([_1, _3], axis=2)
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(_1_3, self.context_mask)
        """


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.1f, smoothed loss %.2f, grad norm %.3f, param norm %.1f, batch time %.2f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))


# Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()

class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
        print("QA Transformer Model created")


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))


# Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
