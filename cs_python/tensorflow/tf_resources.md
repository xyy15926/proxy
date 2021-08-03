---
title: TensorFlow资源管理
categories:
 - Python
 - TensorFlow
tags:
 - Python
 - TensorFlow
 - Resources
date: 2019-08-19 17:25
updated: 2021-08-02 15:08:50
toc: true
mathjax: true
comments: true
description: TensorFlow资源管理
---

##	Resources

###	`tf.placeholder`

*placeholder*：占位符，执行时通过`feed_dict`参数设置值

```python
tf.placeholder(
	dtype,
	shape=None,
	name=None
)
```

> - `shape`：并不必须，但最好设置参数方便debug

-	需要导入数据给placeholder，可能影响程序速度
-	方便用户替换图中值

###	`tf.data.DataSet`

```python
class tf.data.DataSet:
	def __init__(self)

	# 从tensor slice创建`Dataset`
	def from_tensor_slices(self,
		?data=(?features, ?labels)
	):
		pass

	# 从生成器创建`Dataset`
	def from_generator(self,
		gen,
		output_types,
		output_shapes
	):
		pass

	# 迭代数据集一次，无需初始化
	def make_one_shot_iterator(self):
		pass

	# 迭代数据集任意次，每轮迭代需要初始化
	def make_initializable_iterator(self):
		pass

	# shuffle数据集
	def shuffle(self, ?seed:int):
		pass

	# 重复复制数据集
	def repeat(self, ?times:int):
		pass

	# 将数据集划分为batch
	def batch(self, batch_size:int):
		pass

	def map(self, func:callable):
		pass
```

-	创建只能迭代一轮的迭代器

	```python
	iterator = dataset.make_one_shot_iterator()
	# 这里`X`、`Y`也是OPs，在执行时候才返回Tensor
	X, Y = iterator.get_next()

	with tf.Session() as sess:
		print(sess.run([X, Y]))
		print(sess.run([X, Y]))
		print(sess.run([X, Y]))
			# 每次不同的值
	```

-	创建可以多次初始化的迭代器

	```python
	iterator = data.make_initializable_iterator()
	with tf.Session() as sess:
		for _ in range(100):
			# 每轮重新初始化迭代器，重新使用
			sess.run(iterator.initializer)
			total_loss = 0
			try:
				while True:
					sess.run([optimizer])
			# 手动处理迭代器消耗完毕
			except tf.error.OutOfRangeError:
				pass
	```

-	`tf.data`和`tf.placeholder`适合场景对比
	-	`tf.data`速度更快、适合处理多数据源
	-	`tf.placeholder`更pythonic、原型开发迭代速度快

####	读取文件数据

可以从多个文件中读取数据

```python
 # 文件每行为一个entry
class tf.data.TextLineDataset(filenames):
 # 文件中entry定长
class tf.data.FixedLengthRecordDataset(filenames):
 # 文件为tfrecord格式
class tf.data.TFRecordDataset(filenames):
```

####	`tf.data.Iterator`

```python
class tf.data.Iterator:
	# 获取下组迭代数据
	def get_next():
		pass

	# 根据dtype、shape创建迭代器
	@classmethod
	def from_structure(self,
		?dtype: type,
		?shape: [int]/(int)
	):
		pass

	# 从数据中初始化迭代器
	def make_initializer(self,
		?dataset: tf.data.Dataset
	):
		pass
```

