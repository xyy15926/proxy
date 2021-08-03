---
title: TensorFlow 持久化
categories:
 - Python
 - TensorFlow
tags:
 - Python
 - TensorFlow
 - Machine Learning
 - Persistence
date: 2019-08-19 17:13
updated: 2019-08-19 17:13
toc: true
mathjax: true
comments: true
description: TensorFlow持久化
---

##	Session Checkpoint

```python
class tf.train.Saver:
	def __init__(self,
		var_list=None/list/dict,
		reshape=False,
		sharded=False,
		max_to_keep=5,
		keep_checkpoint_every_n_hours=10000.0,
		name=None,
		restore_sequentially=False,
		saver_def=None,
		builder=None,
		defer_build=False,
		allow_empty=False,
		write_version=tf.train.SaverDef.V2,
		pad_step_number=False,
		save_relative_paths=False,
		filename=None
	):
		self.last_checkpoints
```

-	用途：保存Session中变量（张量值），将变量名映射至张量值

-	参数
	-	`var_list`：待保存、恢复变量，缺省所有
		-	变量需在`tf.train.Saver`实例化前创建
	-	`reshape`：允许恢复并重新设定张量形状
	-	`sharded`：碎片化保存至多个设备
	-	`max_to_keep`：最多保存checkpoint数目
	-	`keep_checkpoint_every_n_hours`：checkpoint有效时间
	-	`restore_sequentially`：各设备中顺序恢复变量，可以
		减少内存消耗

-	成员
	-	`last_checkpoints`：最近保存checkpoints

###	保存Session

```python
def Saver.save(self,
	sess,
	save_path,
	global_step=None/str,
	latest_filename=None("checkpoint")/str,
	meta_graph_suffix="meta",
	write_meta_graph=True,
	write_state=True
) -> str(path):
	pass
```

-	用途：保存Session，要求变量已初始化

-	参数
	-	`global_step`：添加至`save_path`以区别不同步骤
	-	`latest_filename`：checkpoint文件名
	-	`meta_graph_suffix`：MetaGraphDef文件名后缀

###	恢复Session

```python
def Saver.restore(sess, save_path(str)):
	pass
```

-	用途：从`save_path`指明的路径中恢复模型

> - 模型路径可以通过`Saver.last_checkpoints`属性、
	`tf.train.get_checkpoint_state()`函数获得

###	`tf.train.get_checkpoint_state`

```c
def tf.train.get_checkpoint_state(
	checkpoint_dir(str),
	latest_filename=None
):
	pass
```

-	用途：获取指定checkpoint目录下checkpoint状态
	-	需要图结构已经建好、Session开启
	-	恢复模型得到的变量无需初始化

```python
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
saver.restore(ckpt.model_checkpoint_path)
saver.restore(ckpt.all_model_checkpoint_paths[-1])
```

##	Graph Saver

###	`tf.train.write_graph`

```python
def tf.train.write_graph(
	graph_or_graph_def: tf.Graph,
	logdir: str,
	name: str,
	as_text=True
)
```

-	用途：存储图至文件中

-	参数
	-	`as_text`：以ASCII方式写入文件

##	Summary Saver

###	`tf.summary.FileWriter`

```python
class tf.summary.FileWriter:
	def __init__(self,
		?path=str,
		graph=tf.Graph
	)

	# 添加summary记录
	def add_summary(self,
		summary: OP,
		global_step
	):
		pass

	# 关闭`log`记录
	def close(self):
		pass
```

-	用途：创建`FileWriter`对象用于记录log
	-	存储图到**文件夹**中，文件名由TF自行生成
	-	可通过TensorBoard组件查看生成的event log文件

-	说明
	-	一般在图定义完成后、Session执行前创建`FileWriter`
		对象，Session结束后关闭

##	实例

```python
 # 创建自定义summary
with tf.name_scope("summaries"):
	tf.summary.scalar("loss", self.loss)
	tf.summary.scalar("accuracy", self.accuracy)
	tf.summary.histogram("histogram loss", self.loss)
	summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# 从checkpoint中恢复Session
	ckpt = tf.train.get_check_state(os.path.dirname("checkpoint_dir"))
	if ckpt and ckpt.model_check_path:
		saver.restore(sess, ckpt.mode_checkpoint_path)

	# summary存储图
	writer = tf.summary.FileWriter("./graphs", sess.graph)
	for index in range(10000):
		loas_batch, _, summary = session.run([loss, optimizer, summary_op])
		writer.add_summary(summary, global_step=index)

		if (index + 1) % 1000 = 0:
			saver.save(sess, "checkpoint_dir", index)

 # 关闭`FileWriter`，生成event log文件
write.close()
```




