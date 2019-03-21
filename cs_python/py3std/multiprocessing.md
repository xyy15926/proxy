#	Python多线程

##	

Python使用全局解释器锁（GIL），将进程中的线程序列化，即
多线程实际上无法利用多核cpu并行计算，必须使用多进程才能实现
对多核cpu的利用

-	IO密集可以使用多线程
-	CPU密集则必须使用多进程

注意事项

-	win下启动多进程必须在`__main__`模块下

##	`Process`

`Multiprocessing`核心模块，类似于`Thread`，实现多进程创建、
启动、关闭

```python
class Process():
	def __init__(self,
		group,
		target(func),
		name=None/str,
		args=None/(),
		kwargs=None/{}
	):
		self.authkey
		self.daemon
		self.exitcode
		self.name
		self.pid

	def is_alive(self):
		pass

	def join(self,
		timeout=None/num
	):
		pass

	def start(self):
		pass

	def run(self):
		pass
	# `start`方法调用`run`方法，若进程实例未传入`target`，
		# `start`默认执行`run`方法

	def terminate():
		pass
	# 立即结束进程
```


```python
from multiprocessing import Process
import os

def test(name):
	print("Process ID: %s" % (os.getpid())
	print("Parent Process ID: %s" % (os.getppid()))

if __name__ == "__main__":
	proc = Process(target=test, args=("nmask",))
	proc.start()
	proc.join()
```

##	`Pool`

创建管理进程池，子进程多且需要控制子进程数量时可以使用

-	提供指定数量进程供用户调用
-	有新请求提交到pool中时，如果pool没有满，将创建新进程执行
	请求，否则请求等待直到池中有进程结束，然后才会创建新进程
	执行

```python
class Pool:
	def __init__(self,
		processes=None/int,
		initializer=None,
		initargs=(),
		maxstacksperchild=None/int,
		context=None
	):

```

##	`Queue`

控制进程安全，同线程中`Queue`

##	`Pipe`

管理管道

##	`Manager`

常与`Pool`模块一起使用，共享资源，不能使用`Queue`、`Array`

##	`Lock`

避免多进程访问共享资源冲突

##	`Semaphore`

控制对共享资源的访问数量

##	`Event`

实现进程间同步通信

##	Experience

-	子进程报错直接死亡，错误信息默认不会输出到任何地方，所以
	子进程中，多使用`try catch`

