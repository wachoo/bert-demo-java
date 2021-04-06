# encoding=utf-8
import os
import shutil
import sys

import tensorflow as tf

import modeling


def load_model_meta_ckpt(ckpt_path):
    sess = tf.Session()

    bert_config = modeling.BertConfig.from_json_file('chinese_L-12_H-768_A-12/bert_config.json')

    input_ids = tf.placeholder(tf.int32, shape=[None, 128], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, 128], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, 128], name='segment_ids')

    with sess.as_default():
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [6, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [6], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph(meta_path)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
    return sess


def save_model(session, export_dir):
    list_tensor(session.graph)
    inputs = {
        "input_ids":
            session.graph.get_tensor_by_name("input_ids:0"),
        "input_mask":
            session.graph.get_tensor_by_name("input_mask:0"),
        "segment_ids":
            session.graph.get_tensor_by_name("segment_ids:0"),
    }
    outputs = {
        "layer_max": session.graph.get_tensor_by_name("loss/Softmax:0"),
    }
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    tf.saved_model.simple_save(
        session,
        export_dir,
        inputs,
        outputs,
    )


def list_tensor(graph):
    for n in graph.as_graph_def().node:
        print(n.name)


if __name__ == "__main__":
    pretrain_path = sys.argv[1]
    save_path = pretrain_path + "/savedmodel"
    meta_path = pretrain_path + '/model.ckpt-20000.meta'
    ckpt_path = pretrain_path + '/model.ckpt-20000'
    sess = load_model_meta_ckpt(ckpt_path)
    save_model(sess, save_path)
