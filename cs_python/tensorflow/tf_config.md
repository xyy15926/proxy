---
title: TensorFlow 安装配置
categories:
  - Python
  - TensorFlow
tags:
  - Python
  - TensorFlow
  - Configuration
  - CUDA
  - CUDNN
  - NVCC
date: 2019-03-21 17:27:37
updated: 2021-08-04 19:38:46
toc: true
mathjax: true
comments: true
description: TensorFlow安装配置
---

##	安装

###	CUDA、CUDNN、CUDAtookit、NVCC

-	CUDA：compute unified device architecture，通用并行计算
	平台和编程模型，方便使用GPU进行通用计算
-	cuDNN：深度学习加速库，计算设计的库、中间件
	-	C++STL的thrust的实现
	-	cublas：GPU版本blas
	-	cuSparse：稀疏矩阵运算
	-	cuFFT：快速傅里叶变换
	-	cuDNN：深度学习网络加速
-	CUDA Toolkit：包括以下组件
	-	编译器`nvcc`：CUDA-C、CUDA-C++编译器，依赖`nvvm`
		优化器
		（`nvvm`本身依赖`llvm`编译器）
	-	·`debuggers`、`profiler`等工具
	-	科学库、实用程序库
		-	cudart
		-	cudadevrt
		-	cupti
		-	nvml
		-	nvrtc
		-	cublas
		-	cublas_device
	-	示例
	-	驱动：

##	TensorBoard

TensorBoard是包括在TensorFlow中可视化组件

-	运行启动了TB的TF项目时，操作都会输出为事件日志文件
-	TB能够把这些日志文件以可视化的方式展示出来
	-	模型图
	-	运行时行为、状态

```python
$ tensorboard --logdir=/path/to/logdir --port XXXX
```

##	问题

###	指令集

>	Your CPU supports instructions that this TensorFlow binary was not cmpiled to use: SSE1.4, SSE4.2, AVX AVX2 FMA

-	没从源代码安装以获取这些指令集的支持
	-	从源代码编译安装
	-	或者设置log级别
		```python
		import os
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
		import tensorflow as tf
		```
