# -*- coding: utf-8 -*
import tensorflow as tf
import ops
import numpy as np

# import pandas as pd
# this class is used for true system, only predict but no evaluation
class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        embedding_width = model_para['dilated_channels']
        #word_embedding
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    def train_graph(self, is_negsample=False, finetune_suc=False):
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')
        label_seq, self.dilate_input = self.model_graph(self.itemseq_input, train=True, finetune_suc=finetune_suc)

        model_para = self.model_para
        if is_negsample:
            logits_2D = tf.reshape(self.dilate_input, [-1,model_para['dilated_channels']])
            self.softmax_w = tf.get_variable("softmax_w", [model_para['item_size'],  model_para['dilated_channels']],tf.float32,tf.random_normal_initializer(0.0, 0.01))
            self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))
            label_flat = tf.reshape(label_seq, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2* model_para['item_size'])#sample 20% as negatives
            # tf.nn.nce_loss   tf.contrib.nn.rank_sampled_softmax_loss(
            loss =tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D, num_sampled,model_para['item_size'])

            # it does not work
            # num_resampled=int(0.9* num_sampled)
            # loss = tf.contrib.nn.rank_sampled_softmax_loss(weights=self.softmax_w, biases=self.softmax_b, labels=label_flat, inputs=logits_2D, num_sampled=num_sampled,
            #                                                num_resampled=num_resampled,num_classes=np.int64(model_para['item_size']),
            #                                                num_true=1,sampled_values= None ,resampling_temperature=0.1,remove_accidental_hits=True,partition_strategy='mod')
        else:
            logits = ops.conv1d(tf.nn.relu(self.dilate_input), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
            label_flat = tf.reshape(label_seq, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)

        self.loss = tf.reduce_mean(loss)
        self.arg_max_prediction = tf.argmax(logits_2D, 1) #useless, if using negative sampling (i.e., negsample=True), it should be changed such as in predict_graph module

    def model_graph(self, itemseq_input,train=True,finetune_suc=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        label_seq = itemseq_input[:, 1:]
        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   context_seq, name="context_embedding")
        dilate_input = self.context_embedding

        if finetune_suc:
            # prob
            global_step = tf.train.get_or_create_global_step()

            compress_ratio = model_para['compress_ratio']
            for layer_id in range(model_para['succ_layers']):

                replace_rate_prob = self.get_prob(model_para['strategy'], model_para['base_replace_prob'],
                                              global_step, model_para['k'])
                sample_prob = tf.random_uniform(shape=[], minval=0, maxval=1)
                # 替换概率越大，sample_prob大于replace_rate_prob的难度越高，condition越容易变成false，gate偏向1,因此更倾向于用suc替换pre
                condition = sample_prob > replace_rate_prob
                gate = tf.cond(condition, lambda: tf.zeros_like(dilate_input),
                               lambda: tf.ones_like(dilate_input))

                pre_layer_output = dilate_input
                suc_layer_output = ops.nextitnet_residual_block_suc(dilate_input, model_para['suc_dilations'][layer_id],
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
                # print("comress_ratio:{}".format(compress_ratio))
                for offset in range(model_para['compress_ratio']):
                    pre_layer_output = ops.nextitnet_residual_block_rezero(pre_layer_output, model_para['dilations'][layer_id*compress_ratio+offset],
                                                            layer_id*compress_ratio+offset, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
                           #single_layer_implement("",layer_idx*compress_ratio+offset,pre_layer_output)
                    # print("predLayer:{}".format(pre_layer_output))
                layer_output = gate * suc_layer_output + (1. - gate)*pre_layer_output
                dilate_input = layer_output
        else:
            for layer_id in range(model_para['succ_layers']):
                dilate_input = ops.nextitnet_residual_block_suc(dilate_input, model_para['suc_dilations'][layer_id],
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)

        return label_seq, dilate_input



    def predict_graph(self, is_negsample=False, reuse=False,finetune_suc = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        self.input_predict = tf.placeholder('int32', [None, None], name='input_predict')

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False, finetune_suc=finetune_suc)
        model_para = self.model_para

        if is_negsample:
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        else:
            logits = ops.conv1d(tf.nn.relu(dilate_input[:, -1:, :]), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        self.loss_test = tf.reduce_mean(loss)

        probs_flat = tf.nn.softmax(logits_2D)
        # self.g_probs = tf.reshape(probs_flat, [-1, tf.shape(self.input_predict)[1], model_para['item_size']])
        self.g_probs = tf.reshape(probs_flat, [-1, 1, model_para['item_size']])

        # newly added for weishi
        self.top_k = tf.nn.top_k(self.g_probs[:, -1], k=5, name='top-k')

    def get_prob(self, strategy, base_replace_prob, global_step, k):
        if strategy == "constant":
            return base_replace_prob
        else:
            #print global_step
            return tf.cast(global_step, tf.float32)*k+base_replace_prob




















