#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resnet_cifar_train_MMPC_two_phase.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/5/27 06:10 PM   liangcong      1.0   Train ResNet model on CIFAR-10/100 with unbiased constraint loss (two-phase)

run this file with
"python resnet_cifar_train_MMPC_two_phase.py --data_type=cifar_10 --num_blocks=3
--weight_decay=0.0001 --_lambda=2.0 --version=1"
"""
import numpy as np
import tensorflow as tf
import datetime
import os
import utils as uc

from sklearn.metrics import confusion_matrix
from data import get_data_set
from data_aug import torch_input
# from tool import unbiased_constraint as uc
from resnet_cifar_model import model
tf.set_random_seed(1234)
np.random.seed(0)
# ------------------------------command line argument---------------------------------
tf.app.flags.DEFINE_float("_lambda", 3., "constraints on unbiased constrained loss, default: 2.0")
tf.app.flags.DEFINE_float("weight_decay", 0.0001, "weight decay, "
                                                  "(ResNet-20: 0.0001, ResNet-32: 0.0002 ,ResNet-44: 0.0003"
                                                  "ResNet-56: 0.0004, ResNet-110: 0.0005)")
tf.app.flags.DEFINE_string("data_type", 'cifar_10', "cifar_10 or cifar_100, default:cifar_10")
tf.app.flags.DEFINE_integer("top_k", 1, "unbiased_cons loss of top k ")
tf.app.flags.DEFINE_integer("num_blocks", 3, "layers number: 6*num_blocks+2, selected from{3,5,7,9,18}")
tf.app.flags.DEFINE_integer("version", 1, "1 means basic version, 2 means exp() version")
FLAGS = tf.app.flags.FLAGS
_DATA_TYPE = FLAGS.data_type
_WEIGHT_DECAY = FLAGS.weight_decay
_NUM_BLOCKS = FLAGS.num_blocks
_TOP_K = FLAGS.top_k
_LAMBDA = FLAGS._lambda
_VERSION = FLAGS.version
# ------------------------------------------------------------------------------------

# ------------------------------command line argument judge---------------------------
if not isinstance(_NUM_BLOCKS, int):
    print("num_blocks must be a int number")
    os._exit(0)
if not isinstance(_LAMBDA, float):
    print("_lambda must be a float number")
    os._exit(0)
if not isinstance(_DATA_TYPE, str):
    print("data_type must be a string")
    os._exit(0)
if not isinstance(_VERSION, int):
    print("version must be a int number")
    os._exit(0)

if _NUM_BLOCKS not in (3, 5, 7, 9, 18):
    print("num_blocks can only take values from {3,5,7,9,18}")
    os._exit(0)
if _DATA_TYPE not in ('cifar_10', 'cifar_100'):
    print("_DATA_TYPE must be cifar_10 or cifar_100")
    os._exit(0)
if _VERSION not in (1, 2, 3):
    print("version must be selected from {1, 2, 3}, "
          "1 means basic version, 2 means exp() version, 3 means neg log() version")
    os._exit(0)
# ------------------------------------------------------------------------------------

# ------------------------------fixed parameters--------------------------------------
_EPOCH = 160
_BATCH_SIZE = 128
_CLASS_SIZE = int(_DATA_TYPE.split('_')[1])
# ------------------------------------------------------------------------------------

print("data_type: %s, num_layers: %d, _lambda: %f, top k: %d, version: %d"
      % (_DATA_TYPE, 6*FLAGS.num_blocks+2, _LAMBDA, _TOP_K, _VERSION))

# ------------------------------create relative folders-------------------------------
LOAD_PAR_PATH = "./" + _DATA_TYPE + "/" + str(6 * _NUM_BLOCKS + 2) + "/baseline"
LOAD_MODEL_SAVE_PATH = os.path.join(LOAD_PAR_PATH, "model/")
PAR_PATH = "./" + _DATA_TYPE + "/" + str(6 * _NUM_BLOCKS + 2) + "/MMPC/two_phase/version_" \
           + str(_VERSION) + "/top_" + str(_TOP_K) + "/lambda_" + str(_LAMBDA)
_MODEL_SAVE_PATH = os.path.join(PAR_PATH, "model/")

if not os.path.exists(PAR_PATH):
    os.makedirs(PAR_PATH)
if not os.path.exists(_MODEL_SAVE_PATH):
    os.makedirs(_MODEL_SAVE_PATH)
_TENSORBOARD_SAVE_PATH = os.path.join(PAR_PATH, "tensorboard")
# ------------------------------------------------------------------------------------

# ------------------------------data pre-process--------------------------------------
x, y, output, global_step, y_pred_cls, c, phase_train = model(_CLASS_SIZE, _NUM_BLOCKS)

train_x, train_y, train_l = get_data_set(name="train", cifar=_CLASS_SIZE)
test_x, test_y, test_l = get_data_set(name="test", cifar=_CLASS_SIZE)
print("mean subtracted")
mean = np.mean(train_x, axis=1)
train_x = train_x - mean[:, np.newaxis]
test_mean = np.mean(test_x, axis=1)
test_x = test_x - test_mean[:, np.newaxis]
print("mean subtracted end")
epoch_size = len(train_x)
if epoch_size % _BATCH_SIZE == 0:
    steps_per_epoch = epoch_size / _BATCH_SIZE
else:
    steps_per_epoch = int(epoch_size / _BATCH_SIZE) + 1
# ------------------------------------------------------------------------------------

# ------------------------------unbiased-cons loss------------------------------------
MMPC_loss = uc.MMPC_loss(tf.nn.softmax(output), y, _TOP_K, _VERSION)
# ------------------------------------------------------------------------------------

# ------------------------------final loss--------------------------------------------
ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
loss = ce_loss + _LAMBDA * MMPC_loss
# ------------------------------------------------------------------------------------

# ------------------------------weight regularization---------------------------------
t_v = tf.losses.get_regularization_losses()
w12 = tf.add_n([t_v[i] for i in range(len(t_v))])
wl2_loss = w12 * _WEIGHT_DECAY
# ------------------------------------------------------------------------------------

# ------------------------------optimal algorithm setting-----------------------------
if _NUM_BLOCKS == 18:
    boundaries = [steps_per_epoch * _epoch for _epoch in [1, 100, 130]]  # for layer-110
    values = [0.01, 0.1, 0.01, 0.001]
else:
    boundaries = [steps_per_epoch * _epoch for _epoch in [100, 130]]
    values = [0.1, 0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, name='Momentum1', use_nesterov=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = train_op.minimize(loss + wl2_loss, global_step=global_step)
# ------------------------------------------------------------------------------------

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)
tf.summary.scalar("Loss", loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
train_writer = tf.summary.FileWriter(_TENSORBOARD_SAVE_PATH, sess.graph)

model_file = tf.train.latest_checkpoint(LOAD_MODEL_SAVE_PATH)
saver.restore(sess, model_file)
sess.run(global_step.initializer)


def train(num_epoch, f):
    """
    Train CNN
    :param itera: numbers of iteration
    :param f: file for storing the training log
    :return:
    """

    global train_x
    global train_y

    _ce_loss = []
    _MMPC_loss = []
    _test_acc = []
    _cm = []
    _feature = []

    print('start training with MMPC loss (one-phase)')
    print('start training with MMPC loss (one-phase)', file=f)

    print("data_type: %s, num_layers: %d, _lambda: %f, top k: %d, version: %d"
          % (_DATA_TYPE, 6 * FLAGS.num_blocks + 2, _LAMBDA, _TOP_K, _VERSION))
    print("data_type: %s, num_layers: %d, _lambda: %f, top k: %d, version: %d"
          % (_DATA_TYPE, 6 * FLAGS.num_blocks + 2, _LAMBDA, _TOP_K, _VERSION), file=f)

    for i in range(num_epoch):
        print('Epoch: %d' % i)
        print('Epoch: %d' % i, file=f)

        rand_idx = np.arange(epoch_size)
        np.random.shuffle(rand_idx)

        train_x = train_x[rand_idx]
        train_y = train_y[rand_idx]

        for j in range(steps_per_epoch):
            if j < steps_per_epoch - 1:
                batch_xs = train_x[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
                batch_ys = train_y[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
            else:
                batch_xs = train_x[j * _BATCH_SIZE:epoch_size]
                batch_ys = train_y[j * _BATCH_SIZE:epoch_size]

            batch_xs = torch_input(batch_xs)

            i_global, _, l_loss, l_ce_loss, l_MMPC_loss, l_acc = \
                sess.run([global_step, optimizer, loss, ce_loss, MMPC_loss, accuracy],
                         feed_dict={x: batch_xs, y: batch_ys, phase_train: True})

            if i_global % 10 == 0:
                print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, MMPC_loss: %.4f"
                      % (i_global, l_acc, l_loss, l_ce_loss, l_MMPC_loss))
                print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, MMPC_loss: %.4f"
                      % (i_global, l_acc, l_loss, l_ce_loss, l_MMPC_loss), file=f)

            if j == steps_per_epoch - 1:
                data_merged = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, phase_train: True})
                print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, MMPC_loss: %.4f"
                      % (i_global, l_acc, l_loss, l_ce_loss, l_MMPC_loss))
                print("Global step: %d, batch accuracy: %.4f, loss: %.4f, ce loss: %.4f, MMPC_loss: %.4f"
                      % (i_global, l_acc, l_loss, l_ce_loss, l_MMPC_loss), file=f)

                acc, cm, feature = predict_test(f)

                _test_acc.append(acc)
                _ce_loss.append(l_ce_loss)
                _unbiased_cons_loss.append(l_unbiased_cons_loss)

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy/test", simple_value=acc)])
                train_writer.add_summary(data_merged, i_global)
                train_writer.add_summary(summary, i_global)

                if i == 0:
                    saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                    print("Saved checkpoint.")
                    print("Saved checkpoint.", file=f)

                    _temp_acc = acc
                    _temp_feature = feature
                    _temp_cm = cm

                if acc > _temp_acc:
                    saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                    print("Saved checkpoint.")
                    print("Saved checkpoint.", file=f)
                    _temp_acc = acc
                    _temp_feature = feature
                    _temp_cm = cm

    _feature.append(_temp_feature)
    _cm.append(_temp_cm)

    return _ce_loss, _MMPC_loss, _test_acc, _cm, _feature


def predict_test(f, show_confusion_matrix=False):
    """
    Make prediction for all images in test_x
    :param show_confusion_matrix: default false
    :return: accuracy
    """
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    feature = np.zeros(shape=(len(test_x), _CLASS_SIZE), dtype=np.float)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], feature[i:j, :] = sess.run([y_pred_cls, output],
                                                         feed_dict={x: batch_xs, y: batch_ys, phase_train: False})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)), file=f)

    cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc, cm, feature


def main(_):
    f = open(os.path.join(PAR_PATH, "train_info_.txt"), "w")
    time1 = datetime.datetime.now()
    _ce, _MMPC_loss, test_acc, cm, feature = train(_EPOCH, f)
    time2 = datetime.datetime.now()
    duration = time2 - time1
    print("duration:", duration)
    print("duration:", duration, file=f)
    print("the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc)))
    print("the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc)), file=f)
    f.close()

    np.savetxt(os.path.join(PAR_PATH, "ce_loss.txt"), _ce)
    np.savetxt(os.path.join(PAR_PATH, "test_acc.txt"), test_acc)
    np.savetxt(os.path.join(PAR_PATH, "MMPC_loss.txt"), _MMPC_loss)
    np.save(os.path.join(PAR_PATH, "cm.npy"), cm)
    np.save(os.path.join(PAR_PATH, "feature.npy"), feature)


if __name__ == "__main__":
    tf.app.run()

sess.close()

