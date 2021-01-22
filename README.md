# resnet-tf1.13
这是一个使用tensorflow1.13搭建的resnet18。

## 运行程序
```
./run_resnet_64_one.sh

```

## To-do list
- [x] BatchNorm
- [x] L2 Regularization
- [ ] Pre-train model
- [x] parser 参数解释器
- [x] tensorboard
- [x] tensorflow 计算AUC
- [x] tensorflow 计算ACC
- [ ] 多卡并行训练
- [x] tf.train.Saver 保存

##  各个文件的功能
#### 如果想训练自己的resnet只需要以下4个文件。
- dataProcess.py   
```
> dataGenerator.
> 使用dataProcess时，需要按照类别标签建立文件夹，将对应的图片放在该路径下，函数会自动对其划分为训练集和验证集。
> 测试集需要自行提前留出。

```
- resnet.py       
> resnet的实现

- train_resnet_64_one.py         
> 模型训练定义

- run_resnet_64_one.py          
> 训练

- focal_loss.py
> focal_loss的实现。



