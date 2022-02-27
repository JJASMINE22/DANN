## DANN：Domain adversarial neural network模型的tensorflow2实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Attention](#注意事项) 
4. [训练步骤 How2train](#训练步骤) 

## 所需环境
Python3.7
tensorflow-gpu>=2.0	
tensorflow_datasets==4.4.0	
Numpy==1.19.5	
CUDA 11.0+	
Opencv-contrib-python==4.5.1.48	

## 模型结构
Feature_extractor
![image]

Domain_classifier
![image]

Label_predictor
![image]

## 注意事项
1. DANN结构擅于避免模型过学习 
2. feature_extractor与domain_classifier模块合并构成域分类器
3. feature_extractor与label_predictor模块合并构成样本分类器
4. 通过输入真实数据与抽象数据，输出基于域分类的dc_loss，用于domain_classifier的反向传递
5， 将-dc_loss作用于feature_extractor并反向传递，实现混淆真实、抽象数据特征的效果，实现域分类器的分体式训练
6.	将真实数据输入样本分类器，将lp_loss正常作用于feature_extractor与label_predictor

## 训练步骤
1. 默认使用mnist作为真实样本，svhn作为抽象样本
2. 首次运行将自行下载以上两种数据集
3. 运行train.py即可开始训练

