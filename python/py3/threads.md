#	Python多线程

##	`threading`

`threading`可以在单独的线程中执行**任何**在python中可以
**调用**的对象

###	`Thread`

可控线程类，有两种方法使用
-	传递callable创建新对象
-	继承、覆盖`run`方法

```python
from threading import Thread
class Thread:

	def __init__(self,
		group=None,
		target=callable,
		name=None/str,
		args=(),
		kwargs={},
		*,
		daemon=None/True/daemon):
		pass

		# `group`：保留参数用于未来扩展
		# `target`：可执行对象，将被`run` invoke
		# `args`：用于invoke `target`参数tuple
		# `kwargs`：用于invoke `target` keyword参数dict
		# `daemon`：是否启用守护进程，或指定特定的守护进程
			# python解释器会保持运行直到所有线程终止，对应
			# 需要长时间运行的任务、线程，应当使用后台线程

	def is_alive(self):
		pass

		# 返回线程是否存活

	def join(self,
		timeout=None/float):
		pass

		# 等待直到线程结束（“join”：将线程加入当前线程）
			# 可以多次调用
			# 试图导致死锁时，将会raise RuntimeError

		# `timeout`：指定timeout时间，单位秒，否则阻塞直到
			# 线程结束

	def run(self):
		pass

		# 代表线程活动
			# 覆盖此方法设置线程活动
			# 原`run`用于invoke `target`

	def setDaemon(self, daemonic):
		pass

	def setName(self, name):
		pass

	def start(self):
		pass

		# 开始线程活动（线程创建完成后不会立即执行）
		# 多次调用raise RuntimeError
```

###	终止线程

####	轮询方法

```python
from threading import Thread
import time

class CountDownTask:

	def __init__(self):
		self._running = True

	def terminate(self):
		self._runing = False

	def run(self, n):
		while self._running and n > 0:
			# 设置轮询点，告诉线程何时应该终止
			print("T-minus", n)
			n -= 1
			time.sleep(5)

def test():
	c = CountdownTask()
	t = Thread(target=c.run, args=(10,0))
		# 创建Thread
	t.start()
		# 启动线程
	c.terminate()
	c.join()
```

####	超时循环方法

线程执行一些类似IO的阻塞操作，通过轮询终止线程会使线程间的
协调非常棘手，如：线程一直阻塞在IO操作，则其无法返回，也无法
检查自己是否应该结束。此时需要利用超时循环操作线程

```python
class IOTask:
	def terminate(self):
		self._running = False
	
	def run(self, sock):
		sock.settimeout(5)
			# 设置超时
		while self._running:
			try:
				data = sock.recv(8192)
				break
			except socket.timeout:
				continue
			
			# continued processing

		# terminated

		return
```

###	继承

这样会使得代码依赖于`threading`，这样代码只能在线程上下文中
使用，而使用`Thread`对象则可以类似的用于其他上下文中，比如
`multiprocessing`模块中

```python
from threading import Thread

class CountDownThread(Thread):
	def __init__(self, n):
		super().__init__();
		self.n = n
	
	def run(self)"
		while self.n > 0:
			print("T-minus", self.n)
			self.n = -1
			time.sleep(5)

def test():
	c = CountdownThread(5)
	c.start()
```

##	线程同步

###	`Event`

线程独立运行、状态不可预测，如果其他线程需要通过判断其他线程
状态确定下一步操作，同步问题棘手。需要使用`Event`，其包含可
由线程设置的信标，允许线程等待某些事件的发生。

-	`Event`对象信标默认设置为`False`，等待Event对象的线程会
	阻塞直到信标设置为真
-	若有线程设置信标为真，则将唤醒**所有**等待该`Event`对象
	线程
	-	如果只想唤醒单个线程，使用信号量、`Condition`代替
-	`Event`对象最好单次使用，即一旦其信标设置为真应该立刻
	丢弃
	-	`.clear`可以重中`Event`对象，但是很难确保**安全**
		清理、重新赋值，可能错过事件、死锁
	-	无法保证重置`Event`对象的代码会在线程再次等待其之前
		执行
	-	如果线程需要不停的重复使用`Event`对象，最好使用
		`Condition`对象代替


```python
from threading import Thread, Event
import time

def countdown(n, start_evt):
	print("countdown starting")
	started_evt.set()
		# 设置信标为`True`
	while n > 0:
		print("T-minus", n)
		n -= 1
		time.sleep(5)

def test():
	start_evt = Event()
	print("Launching countdown")
	t = Thread(target=countdown, args=(10, started_evt))
	t.start()

	start_evt.wait()
		# 等待`start_evt`，阻塞直到其信标设置为`True`
	print("countdown is running")
```

###	`Condition`

```python
import threading import Thread, Condition
import time

class PeriodicTimer:
	def __init__(self, interval):
		self._interval = interval
		self._flag = 0
		self._cv = Condition()

	def start(self):
		t = Thread(target=self.run)
		t.deamon = True
		t.start()

	def run(self):
		while True:
			time.sleep(self._interval)
			with self._cv:
				self._flag ^= 1
				self._cv.notify_all()

	def wait_for_tick(self):
		with self._cv:
			# 等待timer的下个tick
			last_flag = self._flag
			while last_flag == self._flag:
				self._cv.wait()

def countdown(nticks, ptimer):
	while nticks > 0:
		ptimer.wait_for_tick()
		print("T-minus", nticks)
		nticks -= 1

def countup(last, ptimer):
	n = 0
	while n < last:
		ptimer.wait_for_tick()
		print("counting", n)
		n += 1

def test():
	ptimer = PeriodTimer(5)
	ptimer.start()
	threading.Thread(target=countdown, args=(10, ptimer)).start()
	threading.Thread(target=countup, args=(5, ptimer)).start()
```

###	`Semaphore`

```python
class Semaphore(builtins.object):
	def __init__(self, value):
		pass
	
	# `value`：起始许可证数量（最多允许同时执行线程数目）

	def acquire(self,
		blocking=True,
		timeout=None):
		pass

		# 获取信号量，内部计数（许可证）`-1`
			# 内部计数`>0`，`-1`立即返回`True`
			# 内部计数`<=0`，阻塞直到其他线程调用`release`
				# 使内部计数`>0`
				# 中途可能被`release`选中结束阻塞
		# 成功获取许可证数量返回`True`

	def release():
		pass

		# 释放信号量，内部计数（许可证）`+1`
		# 内部计数`<0`，表明有些线程阻塞，**随机**选择线程
			# 继续执行
```

信号量对象是建立在共享计数器基础上的同步原语

-	计数器不为0，`with`语句将计数器-1，线程继续执行
-	计数器为0，`with`阻塞线程直到其他线程将计数器+1
-	可以像标准锁一样是使用信号量作线程同步，但会增加复杂性
	影响性能，更适用于需要在线程间引入信号、限制的程序

```python
from threading import Thread, Semaphore

def worker(n, sema):
	sema.acquire()
	print("working", n)

def test():
	sema = Semaphore(0)
	nworkers = 10
	for n in range(nworkers):
		t = Thread(target=worker, args=(n, sema))
		t.start()
		# 创建线程池并启动，但是都等待获取信号量

	sema.release()
	# 释放信号量，第一个线程“启动”（已经启动，只是在阻塞
		# 等待获取信号量
	sema.release()
	# 释放信号量，第二个线程“启动”
```

##	线程间通信

###	`queue`

```python
class Queue(builtins.object):

	def __init__(self, maxsize=0):
		pass
		# `maxsize`；限制可以添加到队列中的元素数量

	def empty(self):
		pass
		# 预期被移除，使用`qsize() == 0`判断队列是否空
		# 和`qsize()`一样非线程安全，在其使用其结果时队列
			# 状态可能已经改变

	def full(self):
		pass
		# 预期被移除，使用`qsize() > n`替代，非线程安全

	def get(self,
		block=True,
		timeout=None/num)

		# remove、return队列中的一个元素
		# `block`
			# `True`：`timeout`延迟时间内默认阻塞直到获取元素
			# `False`：立刻能获取元素，否则raise Empty exception
		# `timeout`：延迟时间，默认无限

	def get_nowait(self):
		pass

		# **非阻塞**的remove、return队列中元素
		# 等同于`get(block=False)`，无法立刻获取元素将
			# raise Empty exception

	def join(self):
		pass

		# 阻塞直到队列中所有元素被获取（消费）、**处理**

	def put(self,
		item,
		block=True,
		timeout=None/num):
		pass

		# 向队列中追加元素
		# `block`
			# `False`：阻塞`timeout`直到有空闲slot可以放入
				元素，
			# `True`：没有空闲slot立即raise Empty exception

	def put_nowait(self, item):
		pass

	def qsize(self):
		pass

		# 返回队列大概大小（不可靠）

	def task_done(self):
		pass

		# 指示前一个队列“任务”完成
		# 队列消费者每次`get`元素，调用`task_done`告诉队列
			# 任务已完成，这任务是指**自定义**任务，不是指
			# 从队列中获取元素“任务”
		# `task_done`需要自行调用，直到所有队列中元素都对应
			# 调用`task_done`，才能解除`join`阻塞
```

线程之间发送数据最安全的方式就是使用`queue.Queue`，创建被
多个线程共享的`Queue`对象

-	`Queue`对象已经包含必要的锁，可以在多个线程间安全的共享
	数据
-	使用`Queue`时，协调生产者、消费者的关闭问题可能会由麻烦
	-	通用做法时在队列中放置特殊的值，消费者读到时终止执行
	-	有多个消费者时，消费者读取特殊值之后可以放回到队列中
		传递下去，关闭所有消费者
-	向队列中添加数据不会复制此数据项，线程间通信实际上是在
	线程间传递对象引用
	-	如果担心对象共享状态，可以传递不可修改数据结构、
		深拷贝

```python

from queue import Queue
from threading import Thread, Event

_sentinel = object()
	# object signaling shutdown

def producer(out_q):
	# thread producing data
	while True:
		evt = Event()
		out_q.put((data, evt))
		evt.wait()
			# producer监听`evt`，在消费者处理完特定数据后
				# 得到通知
	evt = Event()
	out_q.put((_sentinel, evt))

def consumer(in_q):
	# thread consuming data
	while True:
		data, evt = in_q.get()
		evt.set()
			# 设置信标通知producer
		if data is _sentinel:
			evt.clear()
			in_q.put((_sentinel, evt))
				# 读取后放回对象传递下去
			break

def test():
	q = Queue()
	t1 = Thread(target=consumer, args=(q,))
	t2 = Thread(target=producer, args=(q,))
	t1.start()
	t2.start()

	q.join()
		# 阻塞直到队列清空，所有元素被消耗
```

###	自定义

####	`Condition`

最常见方法时使用`Condition`变量包装数据结构

```python
import heapq
from threading import Condition

class PriortyQueue:
	def __init__(self):
		self._queue = [ ]
		self._count = 0
		self._cv = Condition()

	def put(self, item, priority):
		with self._cv:
			heapq.heappush(self._queue, (-priority, self._count, item))
			self._count += 1
			self._cv.notify()

	def get(self):
		with self._cv:
			while len(self._queue) == 0:
				self._cv.wait()
			return heapq.heappop(self._queue)[-1]
```

###	`thread.local`

```python
from socket import socket, AF_INET, SOCK_STREAM
import threading

class LazyConnection:
	def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
		self.address = address
		self.family = family
		self.type = type
		self.local = threading.loca()

	def __enter__(self):
		if hasattr(self.local, "sock"):
			raise RuntimeError("already connected")
			self.local.sock = socket(self.family, self.type)
			self.local.sock.connect(self.address)
			return self.local.sock

	def __exit__(self, exc_ty, exc_val, tb):
		self.local.sock.close()
		del self.local.sock
```

##	锁

对多线程程序中的临界区加锁避免竞争条件

###	关键部分加锁

####	`threading.Lock`

`Lock`对象和`with`语句块一起使用可以保证互斥执行

-	`with`语句会在代码块执行前自动获取锁，执行结束后自动释放

-	每次只能有一个线程可以执行`with`语句包含的代码块

-	也可以使用`Lock().acquire()`、`Lock().release()`显式
	获取、释放锁，但是可能会出现忘记`release`、获取锁之后
	产生异常，使用`with`语句可以保证依然能够正确释放锁

-	为了避免出现死锁，每个线程应该一次只允许获取一个锁，
	否则应该使用更高级死锁避免机制

线程调度本质上是不确定的，在多线程中错误使用锁机制可能会导致
随机数据损坏、其他异常行为，即**竞争条件**，为此最好只在
临界区（对临界资源进行操作的部分代码）使用锁

```python
from threading import Lock

class SharedCounter:
	def __init__(self, initial_value=0):
		self._value = initial_value
		self._value_ok = Lock()

	def incr(self, delta=1):
		with self._value_lock:
			# `with`在会在代码块执行前、后自动获取、释放锁
			# 保证每次只有一个线程可以执行`with`语句
			self._value += delta

	def incr_explicitly(self, delta=1):
		self._value_ok.acquire()
			# 显式获取锁
		self._value += delta
		self._value_ok.release()
			# 显式释放锁

	def decr(self, delta=1):
		with self._value_lock:
			self._value -= delta

	def decr_explicitly(self, delta=1):
		self._value_ok.acquire()
		self._value -= delta
		self._value_ok.release()
```

####	`threading.RLock`

`RLock`（可重入锁）可以被同一个线程多次获取，用于实现基于
检测对象模式的锁定、同步

-	当锁被持有时，只有一个线程可以使用完整的函数/方法
-	与标准锁不同的是，已经持有这个锁的方法调用**使用这个锁**
	的方法时，无需再次获取锁

```python
from threading import RLock

class ShareCounter:
	_lock = RLock()
		# 没有对每个实例中的可变对象加锁，而是一个被所有实例
			# 共享的类级锁
		# 无论类有多个实例都只要一个锁
			# 需要大量使用计数器情况下内存效率更高
			# 但使用大量线程并频繁更新计数器时会有争用锁问题
	def __init__(self, initial_value=0):
		self._value = initial_value

	def incr(self, delta=1):
		with ShareCounter._lock:
			self._value += delta

	def decr(self, delta = 1):
		with SharedCounter._lock:
			# 已经获取锁`_lock`，调用同样使用这个锁的方法
				# `decr`时，无需再次获取锁
			self.incr(-delta)
```

####	`threading.Semaphore`

使用信号量限制一段代码的并发访问量

```python
from threading import Semaphore
import urllib.request

def fetch_url(url, sema):
	with sema:
		return urllib.request.urlopen(url)

def test():
	_fetch_url_sema = Semaphore(5)
```

###	防止死锁

线程需要一次获取多个锁，需要避免死锁问题

####	`contextlib.contextmanager`

```python
from threading import local
from contextlib import contextmanager

_local = local()
@contextmanager
def acquire(*locks):
	locks = sorted(locks, key=lambda x: id(x))
		# 根据object identifier对locks排序
		# 之后根据此list请求锁都会按照固定顺序获取
	acquired = getattr(_local, "acquired", [ ])
	if acquired and max(id(lock)) for lock in acquired >= id(locks[0]):
		# `_local`中已有锁不能比新锁`id`大，否则有顺序问题
		raise RuntimeError("lock order violation")

	acquired.extend(locks)
	_local.acquired = acquired

	try:
		for lock in locks:
			lock.acquire()
		yield
	finally:
		for lock in reversed(locks):
			lock.release()
		del acquired[-len(locks):]

def thread_1(x_lock, y_lock):
	while True:
		with acquire(y_xlock, y_lock):
			print("Thread_1")

def thread_2(x_lock, y_lock):
	while True:
		with acquire(y_lock, x_lock):
			print("Thread_2")

def test():
	x_lock = threading.Lock()
	y_lock = threading.Lock()

	t1 = threading.Thread(target=thread_1, args=(x_lock, y_lock)):
	t1.daemon = True
	t1.start()

	t1 = threading.Thread(target=thread_2, args=(x_lock, y_lock))
	t2.daemon = True
	t2.start()
```
