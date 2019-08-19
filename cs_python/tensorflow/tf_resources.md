---
title: TensorFlow资源管理
tags:
 - Python
 - TensorFlow
 - Machine Learning
 - Resources Managerment
categories:
 - Python
 - TensorFlow
date: 2019-08-19 17:25
updated: 2019-08-19 17:25
toc: true
mathjax: true
comments: true
description: TensorFlow资源管理
---

##	Resources

###	`tf.placeholder`

占位符，之后在真正`.run`时，通过`feed_dict`参数设置值

-	`shape`：并不必须，但最好设置参数方便debug
-	需要导入数据给placeholder，可能影响程序速度

```python
tf.placeholder(
	dtype,
	shape=None,
	name=None
)
```

###	`tf.data`

直接将数据存储在`tf.data.Dataset`对象中

####	`tf.data.Dataset.from_tensor_slice`

-	`features`、`labels`：Tensors或者是ndarray

```python
from_tensor_slice(
	(features, labels)
)
```

####	文件中读取数据

可以从多个文件中读取数据

```python
tf.data.TextLineDataset(filenames)
	# 文件每行为一个entry
tf.data.FixedLengthRecordDataset(filenames)
	# 文件中entry定长
tf.data.TRRecordDataset(filenames)
	# 文件为tfrecord格式
```

####	使用数据

```python
iterator = dataset.make_one_shot_iterator()
	# 创建只能迭代一轮的迭代器
X, Y = iterator.get_next()
	# 这里`X`、`Y`也是OPs，在执行时候才返回Tensor

with tf.Session() as sess:
	print(sess.run([X, Y]))
	print(sess.run([X, Y]))
	print(sess.run([X, Y]))
		# 每次不同的值

iterator = data.make_initializable_iterator()
	# 创建可以多次初始化的迭代器
with tf.Session() as sess:
	for _ in range(100):
		sess.run(iterator.initializer)
			# 每轮重新初始化迭代器，重新使用
		total_loss = 0
		try:
			while True:
				sess.run([optimizer])
		except tf.error.OutOfRangeError:
			# 手动处理迭代器消耗完毕
			pass
```

####	处理数据

```python
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(1000)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x, 10))
```

##	Checkpoint

##	Summary

####	`tf.summary.FileWriter`

-	创建`FileWriter`对象用于记录log
-	存储图到**文件夹**中，文件名由TF自行生成
-	生成event log文件可以通过TensorBoard组件查看

```python
writer = tf.summary.FileWriter("./graphs", g1)
	# 创建`FileWriter`用于记录log
	# 在图定义/构建完成后、会话**执行**图前创建
with tf.Session() as sess:
	# writer = tf.summary.FileWriter("./graphs", sess.graph)
		# 也可以在创建Session之后，记录Session中的图
	session.run(a)
write.close()
	# 关闭`FileWriter`，生成event log文件
```



