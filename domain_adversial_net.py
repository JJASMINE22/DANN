# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     Dense,
                                     Flatten,
                                     LeakyReLU,
                                     Activation,
                                     BatchNormalization,
                                     MaxPooling2D,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np

class DANN():
    """
    主要应用于真实数据量不足的情况下
    通过混淆真实图像与同类抽象图的特征, 提高模型泛化性
    """
    def __init__(self,
                 input_shape,
                 cls_num,
                 lr=[0.001, 0.0001, 0.005],
                 **kwargs):
        """
        :param input_shape: 图像输入形状
        :param cls_num: 真实样本类别数
        :param lr: 学习率
        """
        super(DANN, self).__init__(**kwargs)
        assert isinstance(lr, list)
        self.input_shape = input_shape
        self.cls_num = cls_num
        self.feature_extractor = self.create_feature_extractor()
        self.label_predictor = self.create_label_predictor()
        self.domain_classifier = self.create_domain_classifier()

        self.classify_domain = Sequential([self.feature_extractor,
                                           self.domain_classifier])

        self.predict_label = Sequential([self.feature_extractor,
                                         self.label_predictor])

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.lp_lr = lr[0]
        self.dc_lr = lr[1]
        self.fe_lr = lr[2]

        self.lp_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lp_lr, decay=0.00005)
        self.dc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.dc_lr, decay=0.0002)
        self.fe_optimizer = tf.keras.optimizers.Adam(learning_rate=self.fe_lr, decay=0.0001)

        self.train_lp_loss = tf.keras.metrics.Mean()
        self.train_dc_loss = tf.keras.metrics.Mean()

        self.train_lp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_dc_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        self.test_lp_loss = tf.keras.metrics.Mean()
        self.test_dc_loss = tf.keras.metrics.Mean()

        self.test_lp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_dc_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def create_feature_extractor(self):
        """
        使用贯续模型创建特征提取器
        """
        feature_extractor = Sequential([
            Conv2D(filters=32, kernel_size=5, strides=1, padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            MaxPooling2D(pool_size=3, strides=2, padding='same'),
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            MaxPooling2D(pool_size=3, strides=2, padding='same'),
            # Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
            # BatchNormalization(),
            # LeakyReLU(alpha=0.2),
            # MaxPooling2D(pool_size=3, strides=2, padding='same')
            # Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
            # BatchNormalization(),
            # LeakyReLU(alpha=0.2),
        ])

        return feature_extractor

    def create_label_predictor(self):
        """
        使用贯续模型创建真实样本分类器
        """
        label_predictor = Sequential([
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            GlobalAveragePooling2D(),
            Dense(units=64),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dense(units=self.cls_num),
            Activation('softmax')
        ])

        return label_predictor

    def create_domain_classifier(self):
        """
        使用贯续模型创建真实、抽象样本域对抗器
        """
        domain_classifier = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            GlobalAveragePooling2D(),
            Dense(units=2),
            Activation('sigmoid')
        ])

        return domain_classifier

    def train(self, target, domain):
        """
        :param target: 分类器训练样本
        :param domain: 域对抗器训练样本
        """
        with tf.GradientTape() as tape:
            domain_pred = self.classify_domain(domain[0], training=True)
            dc_loss = self.loss(domain[1], domain_pred)
        dc_grad = tape.gradient(dc_loss, self.domain_classifier.trainable_variables)

        # 与标准DANN略微不同, 此处异步提前优化特征提取器
        with tf.GradientTape() as tape:
            domain_pred = self.classify_domain(domain[0], training=True)
            class_pred = self.predict_label(target[0], training=True)
            lp_loss = self.loss(target[1], class_pred) - self.loss(domain[1], domain_pred)
        fe_grad = tape.gradient(lp_loss, self.feature_extractor.trainable_variables)
        self.fe_optimizer.apply_gradients(zip(fe_grad, self.feature_extractor.trainable_variables))

        # 利用优化后的特征提取器训练分类器, 实验表明训练效果更好
        with tf.GradientTape() as tape:
            class_pred = self.predict_label(target[0], training=True)
            lp_loss = self.loss(target[1], class_pred)
        lp_grad = tape.gradient(lp_loss, self.label_predictor.trainable_variables)
        del tape

        self.dc_optimizer.apply_gradients(zip(dc_grad, self.domain_classifier.trainable_variables))
        self.lp_optimizer.apply_gradients(zip(lp_grad, self.label_predictor.trainable_variables))

        self.train_lp_loss(lp_loss)
        self.train_lp_accuracy(target[1], class_pred)

        self.train_dc_loss(dc_loss)
        self.train_dc_accuracy(domain[1], domain_pred)

    def test(self, target, domain):
        with tf.GradientTape() as tape:
            domain_pred = self.classify_domain(domain[0])
            dc_loss = self.loss(domain[1], domain_pred)

        with tf.GradientTape() as tape:
            class_pred = self.predict_label(target[0])
            lp_loss = self.loss(target[1], class_pred)

        self.test_dc_loss(dc_loss)
        self.test_dc_accuracy(domain[1], domain_pred)

        self.test_lp_loss(lp_loss)
        self.test_lp_accuracy(target[1], class_pred)


if __name__ == '__main__':

    dann = DANN(input_shape=(28, 28, 3), cls_num=10)
