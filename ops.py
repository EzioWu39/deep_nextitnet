import tensorflow as tf
import math
import numpy as np

# config e.g. dilations: [1,4,16,] In most cases[1,4,] is enough
def nextitnet_residual_block(input_, dilation, layer_id,
                             residual_channels, kernel_size,
                             causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1

def nextitnet_residual_block_rezero(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        rez = tf.get_variable('rez', [1],
                               initializer=tf.constant_initializer(0.0))
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        #input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        #input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1*rez

def nextitnet_residual_block_suc(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True):
    resblock_type = "decoder"
    layer_type = "suc"
    resblock_name = "nextitnet_residual_block{}_{}_layer_{}_{}".format(resblock_type, layer_type,layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        rez = tf.get_variable('rez', [1],
                               initializer=tf.constant_initializer(0.0))
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        #input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        #input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1*rez

# encoder-decoer
def nextitnet_residual_block_ED(input_, dilation, layer_id,
                                residual_channels, kernel_size,
                                causal=True, train=True, encoder=True):
    if encoder == True:
        resblock_type = "encoder"
    else:
        resblock_type = "decoder"

    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def get_adapter(input_, hidden_size=64):
    with tf.variable_scope("adapters"):
        conv_down = conv1d(input_, hidden_size,
                           name="adapter_conv1_down"
                           )
        # relu2 = tf.nn.relu(conv1)
        conv_down = gelu(conv_down)

        residual_channels = input_.get_shape()[-1]

        conv_up = conv1d(conv_down, residual_channels,
                         name="adapter_conv1_up"
                         )
        return input_ + conv_up


# Aggregated Residual Transformations for Deep Neural Networks block1
def get_adapter_split_trans_aggr(input_, cardinality=32):
    with tf.variable_scope("adapters"):
        residual_channels = input_.get_shape()[-1]
        # assert  (residual_channels / (cardinality * 2)).isdigit()
        hidden_size = residual_channels / (cardinality * 4)

        blocksets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size,
                                 name="adapter_conv1_down_{}".format(i)
                                 )
            conv_down_i = gelu(conv_down_i)
            conv_up_i = conv1d(conv_down_i, residual_channels,
                               name="adapter_conv1_up_{}".format(i)
                               )
            blocksets.append(conv_up_i)

        output = tf.add_n(blocksets)
        # relu2 = tf.nn.relu(conv1)
        return input_ + output


# Aggregated Residual Transformations for Deep Neural Networks block1
def get_adapter_split_trans_aggr_withname(input_, cardinality=32, name="adapters"):
    with tf.variable_scope(name):
        residual_channels = input_.get_shape()[-1]
        # assert  (residual_channels / (cardinality * 2)).isdigit()
        hidden_size = residual_channels / (cardinality * 4)

        blocksets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size,
                                 name="adapter_conv1_down_{}".format(i)
                                 )
            conv_down_i = gelu(conv_down_i)
            conv_up_i = conv1d(conv_down_i, residual_channels,
                               name="adapter_conv1_up_{}".format(i)
                               )
            blocksets.append(conv_up_i)

        output = tf.add_n(blocksets)
        # relu2 = tf.nn.relu(conv1)
        return input_ + output


# Aggregated Residual Transformations for Deep Neural Networks block2
def get_adapter_split_trans_aggr_concat(input_, cardinality=32):
    with tf.variable_scope("adapters"):
        residual_channels = input_.get_shape()[-1]
        # assert  (residual_channels / (cardinality * 2)).isdigit()
        hidden_size = residual_channels / (cardinality * 4)

        blocksets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size,
                                 name="adapter_conv1_down_{}".format(i)
                                 )

            blocksets.append(conv_down_i)

        concat = tf.concat(blocksets, axis=2)
        conv_up_i = gelu(concat)
        conv_up_i = conv1d(conv_up_i, residual_channels,
                           name="adapter_conv1_up"
                           )

        # relu2 = tf.nn.relu(conv1)
        return input_ + conv_up_i


def nextitnet_residual_block_adapter(input_, dilation, layer_id,
                                     residual_channels, kernel_size,
                                     causal=True, train=True, adapter=True, hidden_size_down=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter(relu1, hidden_size_down)
            return input_ + after_adapter
        else:
            return input_ + relu1


# very bad when use parallel for nextitnet
def nextitnet_residual_block_adapter_aggr_parallel(input_, dilation, layer_id,
                                                   residual_channels, kernel_size,
                                                   causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter_split_trans_aggr(input_, cardinality)
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            sumops = tf.add(relu1, after_adapter)
            return input_ + sumops


def nextitnet_residual_block_adapter_aggr_beforelayernorm_parallel(input_, dilation, layer_id,
                                                                   residual_channels, kernel_size,
                                                                   causal=True, train=True, adapter=True,
                                                                   cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter_split_trans_aggr(input_, cardinality)
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv = tf.add(dilated_conv, after_adapter)

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return relu1 + input_


# two adapters
def nextitnet_residual_block_2adapter_aggr_beforelayernorm_parallel(input_, dilation, layer_id,
                                                                    residual_channels, kernel_size,
                                                                    causal=True, train=True, adapter=True,
                                                                    cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(input_, cardinality, name="adapters_1")
            dilated_conv = tf.add(dilated_conv, after_adapter)

        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(relu1, cardinality, name="adapters_2")
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv = tf.add(dilated_conv, after_adapter)

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return relu1 + input_


# about 0.5% worse than before layernorm
def nextitnet_residual_block_2adapter_aggr_parallel(input_, dilation, layer_id,
                                                    residual_channels, kernel_size,
                                                    causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(input_, cardinality, name="adapters_1")
            dilated_conv = tf.add(input_ln, after_adapter)
        relu1 = tf.nn.relu(dilated_conv)

        dilated_conv_2 = conv1d(relu1, residual_channels,
                                2 * dilation, kernel_size,
                                causal=causal,
                                name="dilated_conv2"
                                )

        input_ln = layer_norm(dilated_conv_2, name="layer_norm2", trainable=train)

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(relu1, cardinality, name="adapters_2")
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            after_adapter = tf.add(input_ln, after_adapter)

        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(after_adapter)
        return relu1 + input_


# performance is very bad since adaapter is after ReLu
def nextitnet_residual_block_2adapter_aggr_before_ReLu_parallel(input_, dilation, layer_id,
                                                                residual_channels, kernel_size,
                                                                causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?

        relu1 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(input_, cardinality, name="adapters_1")
            dilated_conv = tf.add(relu1, after_adapter)

        dilated_conv_2 = conv1d(dilated_conv, residual_channels,
                                2 * dilation, kernel_size,
                                causal=causal,
                                name="dilated_conv2"
                                )

        input_ln = layer_norm(dilated_conv_2, name="layer_norm2", trainable=train)
        relu2 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(dilated_conv, cardinality, name="adapters_2")
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv_2 = tf.add(relu2, after_adapter)

        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)

        return dilated_conv_2 + input_


def nextitnet_residual_block_adapter_aggr(input_, dilation, layer_id,
                                          residual_channels, kernel_size,
                                          causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter_split_trans_aggr(relu1, cardinality)
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            return input_ + after_adapter
        else:
            return input_ + relu1


def nextitnet_residual_block_adapter_aggr_beforelayernorm(input_, dilation, layer_id,
                                                          residual_channels, kernel_size,
                                                          causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter_split_trans_aggr(dilated_conv, cardinality)
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv = after_adapter

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def nextitnet_residual_block_2adapter_aggr_beforelayernorm(input_, dilation, layer_id,
                                                           residual_channels, kernel_size,
                                                           causal=True, train=True, adapter=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(dilated_conv, cardinality, name="adapters_1")
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv = after_adapter

        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter_split_trans_aggr_withname(dilated_conv, cardinality, name="adapters_2")
            # after_adapter = get_adapter_split_trans_aggr_while_loop(relu1, cardinality)
            # after_adapter = get_adapter(relu1, cardinality)
            dilated_conv = after_adapter

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


# build adapter before layernormalization,then you have optimize the layernormalization layer
def nextitnet_residual_block_adapter_beforelaynorm(input_, dilation, layer_id,
                                                   residual_channels, kernel_size,
                                                   causal=True, train=True, adapter=True, hidden_size_down=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter(dilated_conv, hidden_size_down)
            dilated_conv = after_adapter

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


# paralleled adapters
def nextitnet_residual_block_adapter_beforelaynorm_parallel(input_, dilation, layer_id,
                                                            residual_channels, kernel_size,
                                                            causal=True, train=True, adapter=True, hidden_size_down=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if adapter:
            after_adapter = get_adapter(input_, hidden_size_down)
            # dilated_conv= after_adapter
            dilated_conv = tf.add(dilated_conv, after_adapter)

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1


# very bad when use parallel  for nextitnet
def nextitnet_residual_block_adapter_parallel(input_, dilation, layer_id,
                                              residual_channels, kernel_size,
                                              causal=True, train=True, adapter=True, hidden_size_down=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        if adapter:
            after_adapter = get_adapter(input_, hidden_size_down)
            # dilated_conv= after_adapter
            dilated_conv = tf.add(relu1, after_adapter)

        return input_ + dilated_conv


def nextitnet_densenet_block(input_, dilation, block_id,
                             residual_channels, kernel_size,
                             causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_dense_block{}_block_{}_{}".format(resblock_type, block_id, dilation)
    with tf.variable_scope(resblock_name):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu = tf.nn.relu(input_ln)

        return relu


# Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Identity mappings in deep residual networks.
def nextitnet_residual_block_one(input_, dilation, layer_id,
                                 residual_channels, kernel_size,
                                 causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block_one_{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name):
        input_ln = layer_norm(input_, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        conv1 = conv1d(relu1, int(0.5 * residual_channels), name="conv1d_1")
        conv1 = layer_norm(conv1, name="layer_norm2", trainable=train)
        relu2 = tf.nn.relu(conv1)

        dilated_conv = conv1d(relu2, int(0.5 * residual_channels),
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv"
                              )

        dilated_conv = layer_norm(dilated_conv, name="layer_norm3", trainable=train)
        relu3 = tf.nn.relu(dilated_conv)
        conv2 = conv1d(relu3, residual_channels, name='conv1d_2')
        return input_ + conv2


# Conditional Image Generation with PixelCNN Decoders
def nextitnet_residual_block_gatedCNN(input_, dilation, layer_id,
                                      residual_channels, kernel_size,
                                      causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "gatedCNN_{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv"
                              )
        tanh = tf.nn.tanh(dilated_conv)
        gate_conv = conv1d(input_, residual_channels,
                           dilation, kernel_size,
                           causal=causal,
                           name="gate_conv"
                           )

        sigm = tf.nn.sigmoid(gate_conv)
        multi = tf.multiply(tanh, sigm)
        multi = conv1d(multi, residual_channels, name="conv1d_1")

        return input_ + multi


def conv1d(input_, output_channels,
           dilation=1, kernel_size=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias

        return tf.squeeze(out, [1])


# tf.contrib.layers.layer_norm
def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])],
                               initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [int(shape[-1])],
                                initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
