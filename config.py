# -*- coding: UTF-8 -*-
'''
@Project ：DANN
@File    ：config.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
# ===Domain adversarial neural network===

# data_generator
real_dataset = 'mnist'
fake_dataset = 'svhn_cropped'
train_sample_num = 30000
test_sample_num = 6000
batch_size = 32
ckpt_path = '.\\tf_models\\checkpoint'

# training
epoches = 100
input_shape=(32, 32, 3)
cls_num=10
learning_rate = [0.001, 0.0001, 0.005]