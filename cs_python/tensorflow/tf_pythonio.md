---
title: TensorFlow Python IO接口
categories:
 - Python
 - TensorFlow
tags:
 - Python
 - TensorFlow
 - ProtoBuf
 - IO
date: 2019-08-19 17:20
updated: 2019-08-19 17:20
toc: true
mathjax: true
description: TensorFlow Python IO接口
---

##	TFRecord

TFRecord格式：序列化的`tf.train.Example` protbuf对象

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

###	Feature/Features

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

####	示例

-	转换、写入TFRecord

	```python
	# 创建写入文件
	writer = tf.python_io.TFRecord(out_file)
	shape, binary_image = get_image_binary(image_file)
	# 创建Features对象
	featurs = tf.train.Features(
		feature = {
			"label": tf.train.Feature(int64_list=tf.train.Int64List(label)),
			"shape": tf.train.Feature(bytes_list=tf.train.BytesList(shape)),
			"image": tf.train.Feature(bytes_list=tf.train.BytesList(binary_image))
		}
	)
	# 创建包含以上特征的示例对象
	sample = tf.train.Example(features=Features)
	# 写入文件
	writer.write(sample.SerializeToString())
	writer.close()
	```

-	读取TFRecord

	```python
	dataset = tf.data.TFRecordDataset(tfrecord_files)
	dataset = dataset.map(_parse_function)
	def _parse_function(tf_record_serialized):
		features = {
			"labels": tf.FixedLenFeature([], tf.int64),
			"shape": tf.FixedLenFeature([], tf.string),
			"image": tf.FixedLenFeature([], tf.string)
		}
		parsed_features = tf.parse_single_example(tfrecord_serialized, features)
		return parsed_features["label"], parsed_features["shape"],
				parsed_features["image"]
	```

###	其他函数




