# -*- coding: UTF-8 -*-
'''
@Project ：DANN
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import cv2
import numpy as np
import config as cfg
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import ZeroPadding2D
from domain_adversial_net import DANN

def stack_pad_image(x):
    """
    将单通道复制为三通道
    此处既可使用stack转置拼接亦可使用repeat复制
    """
    x = ZeroPadding2D(padding=2)(x)
    x = tf.squeeze(tf.stack([x, x, x], axis=-1)).numpy()

    return x

if __name__ == '__main__':

    # DANN, 使用mnist作用真实样本, svhn作为抽象样本
    ds_svhn_train, ds_svhn_test = tfds.load(cfg.fake_dataset)['train'], tfds.load(cfg.fake_dataset)['test']
    ds_mnist_train, ds_mnist_test = tfds.load(cfg.real_dataset)['train'], tfds.load(cfg.real_dataset)['test']

    ds_svhn_train = ds_svhn_train.shuffle(cfg.train_sample_num).batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_mnist_train = ds_mnist_train.shuffle(cfg.train_sample_num).batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ds_svhn_test = ds_svhn_test.shuffle(cfg.test_sample_num).batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_mnist_test = ds_mnist_test.shuffle(cfg.test_sample_num).batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    dann = DANN(input_shape=cfg.input_shape, cls_num=cfg.cls_num, lr=cfg.learning_rate)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(feature_extractor=dann.feature_extractor,
                               domain_classifier=dann.domain_classifier,
                               label_predictor=dann.label_predictor,
                               fe_optimizer=dann.fe_optimizer,
                               dc_optimizer=dann.dc_optimizer,
                               lb_optimizer=dann.lp_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    for epoch in range(cfg.epoches):

        for bs_svhn, bs_mnist in zip(ds_svhn_train, ds_mnist_train):
            dc_image = np.concatenate([bs_svhn['image'].numpy(), stack_pad_image(bs_mnist['image'])], axis=0) / 255.
            dc_label = np.concatenate([np.zeros_like(bs_svhn['label']), np.ones_like(bs_mnist['label'])], axis=0)
            lp_image = stack_pad_image(bs_mnist['image']) / 255.
            lp_label = bs_mnist['label'].numpy()
            dann.train([lp_image, lp_label], [dc_image, dc_label])

        for bs_svhn, bs_mnist in zip(ds_svhn_test, ds_mnist_test):
            dc_image = np.concatenate([bs_svhn['image'].numpy(), stack_pad_image(bs_mnist['image'])], axis=0) / 255.
            dc_label = np.concatenate([np.zeros_like(bs_svhn['label']), np.ones_like(bs_mnist['label'])], axis=0)
            lp_image = stack_pad_image(bs_mnist['image']) / 255.
            lp_label = bs_mnist['label'].numpy()
            dann.test([lp_image, lp_label], [dc_image, dc_label])

        print(
            f'Epoch {epoch + 1}, '
            f'dc_Loss: {dann.train_dc_loss.result()}, '
            f'dc_Accuracy: {dann.train_dc_accuracy.result() * 100}, '
            f'lp_Loss:  {dann.train_lp_loss.result()}, '
            f'lp_Accuracy: {dann.train_lp_accuracy.result() * 100}, '
            f'Test dc_Loss: {dann.test_dc_loss.result()}, '
            f'Test dc_Accuracy: {dann.test_dc_accuracy.result() * 100}, '
            f'Test lp_Loss: {dann.test_lp_loss.result()}, '
            f'Test lp_Accuracy: {dann.test_lp_accuracy.result() * 100}'
        )

        ckpt_save_path = ckpt_manager.save()

        dann.train_dc_loss.reset_states()
        dann.train_dc_accuracy.reset_states()
        dann.train_lp_loss.reset_states()
        dann.train_lp_loss.reset_states()
        dann.test_dc_loss.reset_states()
        dann.test_dc_accuracy.reset_states()
        dann.test_lp_loss.reset_states()
        dann.test_lp_accuracy.reset_states()
