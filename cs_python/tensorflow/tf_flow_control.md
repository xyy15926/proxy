---
title: TensorFlow控制算符
tags:
 - Python
 - TensorFlow
 - Flow Control
categories:
 - Python
 - TensorFlow
date: 2019-08-19 17:20
updated: 2019-08-19 17:20
toc: true
mathjax: true
description: TensorFlow流程控制运算符
---

##	控制OPs

###	Neural Network Building Blocks

####	`tf.softmax`

####	`tf.Sigmod`

####	`tf.ReLU`

####	`tf.Convolution2D`

####	`tf.MaxPool`

###	Checkpointing

####	`tf.Save`

####	`tf.Restore`

###	Queue and Synchronization

####	`tf.Enqueue`

####	`tf.Dequeue`

####	`tf.MutexAcquire`

####	`tf.MutexRelease`

###	Control Flow

####	`tf.count_up_to`

####	`tf.cond`

`pred`为`True`，执行`true_fn`，否则执行`false_fn`

```python
tf.cond(
	pred,
	true_fn=None,
	false_fn =None,
)
```

####	`tf.case`

####	`tf.while_loop`

####	`tf.group`

####	`tf.Merge`

####	`tf.Switch`

####	`tf.Enter`

####	`tf.Leave`

####	`tf.NextIteration`

