---
title: Tensorflow约定
tags:
  - Python
  - Tensorflow
categories:
  - Python
  - Tensorflow
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Tensorflow约定
---

##	常用参数说明

-	函数书写声明同Python全局

-	以下常用参数如不特殊注明，按照此解释

###	Session

-	`target = ""/str`
	-	含义：执行引擎

-	`graph = None/tf.Graph`
	-	含义：Session中加载的图
	-	默认：缺省为当前默认图

-	`config = None/tf.ConfigProto`
	-	含义：包含Session配置的*Protocal Buffer*
	-	默认：`None`，默认配置

-	`fetches = tf.OPs/[tf.OPs]`
	-	含义：需要获得/计算的OPs值列表
	-	默认：无

-	`feed_dict = None/dict`
	-	含义：替换/赋值Graph中feedable OPs的tensor字典
	-	默认：无
	-	说明
		-	键为图中节点名称、值为向其赋的值
		-	可向所有可赋值OPs传递值
		-	常配合`tf.placeholder`（强制要求）

###	Operators

-	`name = None/str`
	-	含义：Operations名
	-	默认：`None/OP类型`，后加上顺序后缀
	-	说明
		-	重名时TF自动加上`_[num]`后缀

-	`axis = None/0/int`
	-	含义：指定张量轴
	-	默认
		-	`None`：大部分，表示在整个张量上运算
		-	`0`：有些运算难以推广到整个张量，表示在首轴（维）

-	`keepdims=False/True`
	-	含义：是否保持维度数目
	-	默认：`False`不保持

-	`dtype=tf.int32/tf.float32/...`
	-	含义：数据类型
	-	默认：根据其他参数、函数名推断

-	`shape/dims=(int)/[int]`
	-	含义：各轴维数
	-	默认：`None/1`???
	-	说明
		-	`-1`表示该轴维数由TF计算得到
		-	有些情况下，此参数可省略，由TF隐式计算得到，
			但显式指明方便debug

-	`start=int`
	-	含义：起始位置
	-	默认：`0`

-	`stop=int`
	-	含义：终点位置
	-	默认：一般无

##	TensorFlow基本概念

> - TensorFlow将计算的定义、执行分开

###	流程

####	组合计算图

-	为输入、标签创建placeholder
-	创建weigth、bias
-	指定模型
-	指定损失函数
-	创建Opitmizer

####	在会话中执行图中操作

-	初始化Variable
-	运行优化器
-	使用`FileWriter`记录log
-	查看TensorBoard

##	PyTF 模块

###	`tf.nn`

`tf.nn`：神经网络功能支持模块

-	`rnn_cell`：构建循环神经网络子模块

###	`tf.contrib`

`tf.contrib`：包含易于变动、实验性质的功能

-	`bayesflow`：包含贝叶斯计算
-	`cloud`：云操作
-	`cluster_resolver`：集群求解
-	`compiler`：控制TF/XLA JIT编译器
-	`copy_graph`：在不同计算图之间复制元素
-	`crf`：条件随机场
-	`cudnn_rnn`：Cudnn层面循环神经网络
-	`data`：构造输入数据流水线
-	`decision_trees`：决策树相关模块
-	`deprecated`：已经、将被替换的summary函数
-	`distributions`：统计分布相关操作
-	`estimator`：自定义标签、预测的对错度量方式
-	`factorization`：聚类、因子分解
-	`ffmpeg`：使用FFmpeg处理声音文件
-	`framework`：框架类工具，包含变量操作、命令空间、
	checkpoint操作
-	`gan`：对抗生成相关
-	`graph_editor`：计算图操作
-	`grid_rnn`：GridRNN相关
-	`image`：图像操作
-	`input_pipeline`：输入流水线
-	`integrate`：求解常微分方程
-	`keras`：Keras相关API
-	`kernel_methods`：核映射相关方法
-	`kfac`：KFAC优化器
-	`labeled_tensor`：有标签的Tensor
-	`layers`：类似nn里面的函数，经典CNN方法重构
-	`learn`：类似ski-learn的高级API
-	`legacy_seq2seq`：经典seq2seq模型
-	`linalg`：线性代数
-	`linear_optimizer`：训练线性模型、线性优化器
-	`lookup`：构建快速查找表
-	`losses`：loss相关
-	`memory_stats`：设备内存使用情况
-	`meta_graph_transform`：计算图转换
-	`metrics`：各种度量模型表现的方法
-	`nccl`：收集结果的操作
-	`ndlstm`：ndlstm相关
-	`nn`：`tf.nn`某些方法的其他版本
-	`opt`：某些优化器的其他版本
-	`predictor`：构建预测器
-	`reduce_slice_ops`：切片规约
-	`remote_fused_graph`
-	`resampler`：重抽样
-	`rnn`：某些循环神经网络其他版本
-	`saved_model`：更加易用的模型保存、继续训练、模型转换
-	`seq2seq`：seq2seq相关模型
-	`session_bundle`
-	`signal`：信号处理相关
-	`slim`：contrib主模块交互方式、主要入口
-	`solvers`：贝叶斯计算
-	`sparsemax`：稀疏概率激活函数、相关loss
-	`specs`
-	`staging`：分段输入
-	`stat_summarizer`：查看运行状态
-	`statless`：伪随机数
-	`tensor_forest`：可视化工具
-	`testing`：单元测试工具
-	`tfprof`：查看模型细节工具
-	`timeseries`：时间序列工具
-	`tpu`：TPU配置
-	`training`：训练及输入相关工具
-	`util`：Tensors处理相关工具

###	`tf.train`

`tf.train`：训练模型支持

-	优化器
	-	`AdadeltaOptimizer`：Adadelta优化器
	-	`AdamOptimizer`：Adam优化器
	-	`GradientDescentOptimizer`：SGD优化器
	-	`MomentumOptimizer`：动量优化器
	-	`RMSPropOptimizer`：RMSProp优化器

-	数据处理
	-	`Coordinator`：线程管理器
	-	`QueueRunner`：管理读写队列线程
	-	`NanTensorHook`：loss是否为NaN的捕获器
	-	`create_global_step`：创建global step
	-	`match_filenames_once`：寻找符合规则文件名称
	-	`start_queue_runners`：启动计算图中所有队列

-	tfrecord数据
	-	`Example`：tfrecord生成模板
	-	`batch`：生成tensor batch
	-	`shuffle_batch`：创建随机tensor batch

-	模型保存、读取
	-	`Saver`：保存模型、变量类
	-	`NewCheckpointReader`：checkpoint文件读取
	-	`get_checkpoint_state`：从checkpoint文件返回模型状态
	-	`init_from_checkpoint`：从checkpoint文件初始化变量
	-	`latest_checkpoint`：寻找最后checkpoint文件
	-	`list_variable`：返回checkpoint文件变量为列表
	-	`load_variable`：返回checkpoint文件某个变量值

###	`tf.summary`

`tf.summary`：配合tensorboard展示模型信息

-	`FileWriter`：文件生成类
-	`Summary`
-	`get_summary_description`：获取计算节点信息
-	`histogram`：展示变量分布信息
-	`image`：展示图片信息
-	`merge`：合并某个summary信息
-	`merge_all`：合并所有summary信息至默认计算图
-	`scalar`：展示标量值
-	`text`：展示文本信息



