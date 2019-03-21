#	Python内存管理

##	GC机制

-	python的垃圾回收机制基于简单的引用计数

	-	对象引用计数变成0时才会立即删除的
	-	对于循环引用，对象的引用计数永远不可能为0

-	python还有另外的垃圾回收器专门针对循环引用，但是触发机制
	不能信任，手动触发影响代码美观

	```python
	import gc
	gc.collec()
		# 强制启动垃圾回收器
	```

###	`weakref`

通过`weakref`创建弱引用避免循环引用的问题

-	弱引用是一个对象指针，不会增加其引用计数
-	访问弱引用所引用对象，可以/需要像函数一样调用

```python
import weakref

class Node:
	def __init__(self, value):
		self.value = value
		self._parent = None
		self.children = [ ]

	def __repr__(self):
		return "Node({!r})".format(self.value)

	@property
	def parent(self):
		return None if self._parent is None else self._parent()
			# 访问弱引用需要/可以像函数一样调用

	@parent.setter
	def parent(self, node):
		self._parent = weakref.ref(node)
		# 这里建立的是弱引用，如果直接赋值建立强引用，那么
			# 父节点、子节点互相拥有对方的引用，引用计数
			# 永远不能为0

	def add_child(self, child):
		self.children.append(child)
		child.parent = self

def test():
	root = Node("parent")
	c1 = Node("child")
	root.add_child(c1)
	del root
		# 没有对`root`的强引用，可以正常删除
```



