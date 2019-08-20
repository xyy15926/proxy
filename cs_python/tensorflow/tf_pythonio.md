---
title: TensorFlow Python IO接口
tags:
 - Python
 - TensorFlow
 - ProtoBuf
 - IO
categories:
 - Python
 - TensorFlow
date: 2019-08-19 17:20
updated: 2019-08-19 17:20
toc: true
mathjax: true
description: TensorFlow Python IO接口
---

##	TFRecord

TFRecord格式：protocol buffer格式

```python
class tf.python_io.TFRecordWriter:
	def __init__(self,
		?fileanme: str,
		options: tf.python_io.TFRecordOptions,
		name=None
	):
		pass

class tf.python_io.TFRecordReader:
	def __init__(self,
		options: tf.python_io.TFRecordOptions,
		name=None
	):
		pass

	def read(self):
		pass

###	Feature

```python
class tf.train.Features:
	def __init__(self,
		feature: {str: tf.train.Feature}
	):
		pass

class tf.train.Feature:
	def __init__(self,
		int64_list: tf.train.Int64List,
		float64_list: tf.train.Float64List,
	)
```

###	其他函数




