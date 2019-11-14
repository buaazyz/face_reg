## face recognition

一次基于arcface模型的人脸识别的尝试，基本上参考以下的过程

https://github.com/TreB1eN/InsightFace_Pytorch  

训练集用的是 essex大学的数据集

 https://cswww.essex.ac.uk/mv/allfaces/index.html 



1）放弃了采用mtcnn+mxnet的数据清洗，采用了外部的dlib 库，将原始的数据集进行了人脸检测和对齐，将之裁剪为112*112大小的图片

2）重写了基于dataset和dataloader的数据读取部分

3）出于简单起见，重写的eval进行模型的简单评估

4）数据集的划分基于txt进行操作和读取的

5）采用的se-resnet50的模型，预训练数据可在原github上找到





训练结果部分如下所示

![1573701611066](D:\mygithub\face_reg\result\1573701611066.png)