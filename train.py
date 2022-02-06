import cv2
import numpy as np
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
    ds_svhn_train, ds_svhn_test = tfds.load('svhn_cropped')['train'], tfds.load('svhn_cropped')['test']
    ds_mnist_train, ds_mnist_test = tfds.load('mnist')['train'], tfds.load('mnist')['test']

    ds_svhn_train = ds_svhn_train.shuffle(30000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    ds_mnist_train = ds_mnist_train.shuffle(30000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    ds_svhn_test = ds_svhn_test.shuffle(6000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    ds_mnist_test = ds_mnist_test.shuffle(6000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    dann = DANN(input_shape=(32, 32, 3), cls_num=10)

    EPOCHS = 100

    for epoch in range(EPOCHS):

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

        dann.train_dc_loss.reset_states()
        dann.train_dc_accuracy.reset_states()
        dann.train_lp_loss.reset_states()
        dann.train_lp_loss.reset_states()
        dann.test_dc_loss.reset_states()
        dann.test_dc_accuracy.reset_states()
        dann.test_lp_loss.reset_states()
        dann.test_lp_accuracy.reset_states()
