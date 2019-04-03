#	Python内存管理

##	GC机制

-	python的垃圾回收机制基于简单的引用计数

	-	对象引用计数变成0时才会立即删除的
	-	对于循环引用，对象的引用计数永远不可能为0

-	python还有另外的垃圾回收器专门针对循环引用，但是触发机制
	不能信任，手动触发影响代码美观

	```python
	import gc
	gc.collect()
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

##	*Global Intepretor Lock*

全局内存锁：*GIL*，任何python字节码执行前必须获得的解释器锁

-	在任何时刻，只能有一个线程处于工作状态
-	避免多个线程同时操作变量导致内存泄漏、错误释放

###	优势

-	GIL实现简单，只需要管理一把解释器锁就能保证线程内存安全

	-	当然GIL只能保证引用计数正确，避免由此导致内存问题
	-	还需要原子操作、对象锁避免并发更新问题

-	GIL单线程情况下性能更好、稳定，若通过给所有对象引用计数
	加锁来实现线程安全

	-	容易出现死锁
	-	性能下降很多

-	方便兼容C遗留库，这也是python得以发展的原因
	-	很多python需要的C库扩展要求线程安全的内存管理

###	影响

-	Python线程是真正的操作系统线程

	-	在准备好之后必须获得一把共享锁才能运行
	-	每个线程都会在执行一定机器指令和后切换到无锁状态，
		暂停运行
	-	事实上程序在开始时已经在运行“主线程”

	> - 解释器检查线程切换频率`sys.getcheckinterval()`

-	Python线程无法在多核CPU间分配，对CPU-Bound程序基本没有
	提升效果，对于IO-Bound的程序性能仍然有巨大帮助

###	解决方案

-	多进程
	-	进程分支：`os.fork`
	-	派生进程：`multiprocessing.Process`、
		`concurrent.futures`

-	C语言库封装线程：`ctypes`、`cython`

	-	C扩展形式实现任务线程可在python虚拟机作用域外运行
		可以并行运行任意数量线程
	-	在运行时释放GIL、结束后继续运行python代码时重新获取
		GIL，真正实现独立运行

-	使用其他版本Python解释器：只有原始Python版本CPython使用
	GIL实现
	-	Jython
	-	IronPython
	-	PyPy


