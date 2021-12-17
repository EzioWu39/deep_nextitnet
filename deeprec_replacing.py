import tensorflow as tf
import data_pretrain
import generator_deep_replacing
import utils
import time
import math
import numpy as np
import argparse
import collections
import re
import sys
# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

sess = tf.Session()


def optimizer_for_nextitnet_theseus(loss, global_steps,init_lr, betal,finetune_suc=False):
    #global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(init_lr, beta1= betal)
    tvars_update = [var for var in tvars if re.search(r"suc_layer_\d+", var.name)]
    train_op = optimizer.minimize(loss, global_step=global_steps, var_list=tvars_update)
    return train_op


def get_assignment_map_from_checkpoint_for_theseus(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    tvars = tf.trainable_variables()
    with tf.Session() as sess:
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    suc_assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            # fw.write(name)
            # fw.write("\n")
            continue
        if re.search(r"layer_\d+", name):
            suc_layer_var_name = re.sub(r"layer_", "suc_layer_", name)
            if suc_layer_var_name in name_to_variable.keys():
                assignment_map[name] = name_to_variable[name]
                initialized_variable_names[suc_layer_var_name] = 1
                initialized_variable_names[suc_layer_var_name + ":0"] = 1
                suc_assignment_map[name] = name_to_variable[suc_layer_var_name]
                initialized_variable_names[name] = 1
                initialized_variable_names[name + ":0"] = 1
            else:
                assignment_map[name] = name_to_variable[name]
                initialized_variable_names[name] = 1
                initialized_variable_names[name + ":0"] = 1
        else:
            assignment_map[name] = name_to_variable[name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1
    return assignment_map, suc_assignment_map


# Strongly suggest running codes on GPU with more than 10G memory!!!
# if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set, padtoken):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0

        copyseq = list(seq)
        padcount = copyseq.count(padtoken)  # the number of padding elements
        copyseq = copyseq[padcount:]  # the remaining elements
        lenseq_nopad = len(copyseq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        if (lenseq_nopad - 4) < 1:
            subseqtrain.append(seq)
            continue

        for j in range(lenseq_nopad - 4):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [padtoken] * j

            subseq = list(subseqbeg) + list(subseqend)

            # subseq= np.append(subseqbeg,subseqbeg)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)

    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--premodel', type=str, default="model",
                        help='pretrained model path')
    parser.add_argument('--modelpath', type=str, default="model/successor",
                        help='model save path')
    parser.add_argument('--dataset', type=str, default="ml30",
                        help='dataset name')
    parser.add_argument('--datapath', type=str, default="data/movielen_30.csv",
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=3000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=3000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.1,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--padtoken', type=str, default='0',
                        help='is the padding token in the beggining of the sequence')
    parser.add_argument('--k', type=int, default=30000,
                        help='Warm up k steps')
    args = parser.parse_args()

    dataset = args.dataset
    modelpath = args.modelpath
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
        print("create folder {}".format(modelpath))
    else:
        print("folder {} has existed".format(modelpath))

    dl = data_pretrain.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.item
    items = dl.item_dict
    print("len(items)", len(items))

    if args.padtoken in items:
        padtoken = items[args.padtoken]  # is the padding token in the beggining of the sentence
    else:
        # padtoken = sys.maxint
        padtoken = len(items) + 1

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]


    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]


    if args.is_generatesubsession:
        train_set = generatesubsequence(train_set,padtoken)

    model_para = {
        #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items),
        'dilated_channels': 256,
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        # if you find removing residual network, the performance does not obviously decrease, then I think your data does not have strong seqeunce. Change a dataset and try again.
        'dilations': [1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,],
        'suc_dilations': [1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4],
        'kernel_size': 3,
        'learning_rate': 0.0001,
        'batch_size': 256,
        'iterations': 10,
        'compress_ratio': 2,
        'is_negsample': True ,#False denotes no negative sampling
        'strategy': "linear"
    }
    #successor
    succ_layers = len(model_para['dilations'])//model_para['compress_ratio']
    model_para['succ_layers'] = succ_layers
    model_para['base_replace_prob'] = 0.5
    model_para['k'] = (1-model_para['base_replace_prob'])/args.k

    itemrec = generator_deep_replacing.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'], finetune_suc=True)

    #optimization
    global_steps = tf.train.get_or_create_global_step()
    #optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)

    train_op = optimizer_for_nextitnet_theseus(itemrec.loss, global_steps, model_para['learning_rate'],
                                                                  args.beta1)

    itemrec.predict_graph(model_para['is_negsample'], reuse=True, finetune_suc=False)
    #itemrec.predict_graph_finetune(model_para['is_negsample'],reuse=True, finetune_suc = True)

    tf.add_to_collection("dilate_input", itemrec.dilate_input)
    tf.add_to_collection("context_embedding", itemrec.context_embedding)

    # sess= tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # init weight from bert pretrained checkpoints
    # first, build dict_maps
    # Second, initializing global vars
    tvars = tf.trainable_variables()
    var_to_save = [var for var in tvars if not re.search(r"decoder_layer_\d+", var.name)]
    model_dir = args.premodel
    assignment_map, suc_assignment_map = get_assignment_map_from_checkpoint_for_theseus(tvars, model_dir)
    tf.train.init_from_checkpoint(model_dir, assignment_map)
    tf.train.init_from_checkpoint(model_dir, suc_assignment_map)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(var_to_save, max_to_keep=1)


    #numIters = 60000
    numIters = 1
    max_mrr = 0
    max_hit = 0
    save_path = modelpath

    #for iter in range(19, model_para['iterations']):
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            start = time.time()

            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss, results = sess.run(
                [train_op, itemrec.loss,
                 itemrec.arg_max_prediction],
                feed_dict={
                    itemrec.itemseq_input: item_batch
                })
            end = time.time()

            if numIters % 100 == 0:
                print("LOSS: {}\t STEP:{}".format(loss, numIters))

            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------train1")
                print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, train_set.shape[0] / batch_size))
                print("TIME FOR BATCH(s)", end - start)
                print("TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0)

            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------test1")
                if (batch_no + 1) * batch_size < valid_set.shape[0]:
                    # it is well written here when train_set is much larger than valid_set, 'if' may not hold. it will not have impact on the final results.
                    item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run(
                    [itemrec.loss_test],
                    feed_dict={
                        itemrec.input_predict: item_batch
                    })
                print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size))

            batch_no += 1

            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size*1
                curr_preds_5=[]
                rec_preds_5=[] #1
                ndcg_preds_5=[] #1
                curr_preds_20 = []
                rec_preds_20 = []  # 1
                ndcg_preds_20  = []  # 1
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    # if (numIters / (args.eval_iter) < 10):
                    #     if (batch_no_test > 20):
                    #         break
                    # else:
                    # if (batch_no_test > 200):
                    #    break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    [probs] = sess.run(
                        [itemrec.g_probs],
                        feed_dict={
                            itemrec.input_predict: item_batch
                        })
                    for bi in range(probs.shape[0]):
                        pred_items_5 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k)#top_k=5
                        pred_items_20 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k+15)


                        true_item=item_batch[bi][-1]
                        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
                        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                        rank_5=predictmap_5.get(true_item)
                        rank_20 = pred_items_20.get(true_item)
                        if rank_5 ==None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)#2
                            ndcg_preds_5.append(0.0)#2
                        else:
                            MRR_5 = 1.0/(rank_5+1)
                            Rec_5=1.0#3
                            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)#4
                            ndcg_preds_5.append(ndcg_5)  # 4
                        if rank_20 ==None:
                            curr_preds_20.append(0.0)
                            rec_preds_20.append(0.0)#2
                            ndcg_preds_20.append(0.0)#2
                        else:
                            MRR_20 = 1.0/(rank_20+1)
                            Rec_20=1.0#3
                            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                            curr_preds_20.append(MRR_20)
                            rec_preds_20.append(Rec_20)#4
                            ndcg_preds_20.append(ndcg_20)  # 4


                    batch_no_test += 1
                    # print "BATCH_NO: {}".format(batch_no_test)

                mrr = sum(curr_preds_5) / float(len(curr_preds_5))
                mrr_20 = sum(curr_preds_20) / float(len(curr_preds_20))
                hit = sum(rec_preds_5) / float(len(rec_preds_5))
                hit_20 = sum(rec_preds_20) / float(len(rec_preds_20))
                ndcg = sum(ndcg_preds_5) / float(len(ndcg_preds_5))
                ndcg_20 = sum(ndcg_preds_20) / float(len(ndcg_preds_20))

                if mrr > max_mrr:
                    max_mrr = mrr
                    # max_hit = hit
                    print("Save model!  Max mrr_5:", max_mrr)
                    print("Save model!  mrr_20:", mrr_20)
                    print("Save model!  Max hit_5:", hit)
                    print("Save model!  hit_20:", hit_20)
                    print("Save model!  ndcg_5:", ndcg)
                    print("Save model!  ndcg_20:", ndcg_20)

                    saver.save(sess, os.path.join(modelpath, "{}_suc_model_{}_{}".format(dataset, iter, numIters)))
                # print "Accuracy mrr_5:",#5
                # print "Accuracy mrr_20:",   # 5
                # print "Accuracy hit_5:", #5
                # print "Accuracy hit_20:",   # 5
                # print "Accuracy ndcg_5:",   # 5
                # print "Accuracy ndcg_20:",   #
                    #print "curr_preds",curr_preds
                # print "---------------------------Test Accuray----------------------------"
            numIters += 1
            # if numIters % args.save_para_every == 0:
            #     save_path = saver.save(sess,
            #                            "Model_uaf/pretrain/model_pretrain_{}_{}".format(iter, numIters))


if __name__ == '__main__':
    main()
