#	Keras Squential Models

顺序模型：多个网络层的线性堆叠

##	构建模型

```python
class Squential:
	def __init__(self,
		layers([layser]),
	)

	def add(self, layer):
		pass

	def compile(
		optimizer,
		loss,
		metrics
	)
		# 编译，配置学习过程
```

-	模型需要知道输入数据尺寸，所以顺序模型首层需要指定输入
	数据尺寸（后面层可以自动推断）
	-	`input_shape`：表示尺寸的元组，所有Layer支持
	-	`input_dim`：某些2D层支持
	-	`input_length`：某些3D时序层
	-	`batch_size`：指定固定batch大小

```python
class Layer:
	def __init__(self,
		input_shape=None/tuple,
	)
```
