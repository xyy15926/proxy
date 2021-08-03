---
title: 并行开发
categories:
  - Python
  - Cookbook
tags:
  - Python
  - Cookbook
  - Parallel
date: 2019-03-31 23:44:28
updated: 2019-03-31 23:44:28
toc: true
mathjax: true
comments: true
description: 并行开发
---

##	综述

python并行多任务均依赖操作系统底层服务并行执行python代码

-	线程派生：基本所有主流平台均支持
-	多进程
	-	shell命令进程
	-	子python进程

###	跨平台多进程实现

-	创建子python进程
	-	类Unix：`fork`系统调用实现进程分支，分支进程运行时
		环境同主进程完全一致
	-	Win：创建python进程，`import`当前调用代码得到类似
		主进程运行时环境
-	`pickle`序列化被调函数，传递给子进程执行

> - 因为Win下分支进程需要导入当前调用，所以多进程代码必须
	在`__main__`内，否则无限循环创建新进程
> - 进程池调用函数要声明在进程池创建之前，否则启动进程会报错

###	进程通信

-	python进程间数据传递基本都是通过`pickle`序列化传递，所以
	需要传递的数据要支持`pickle`序列化

-	`multiprocessing`等模块派生进程需要传递被调函数，所以
	不支持

	-	`lambda`匿名函数
	-	绑定对象方法

###	其他相关模块、包

-	多进程：`pathos`、`pp`
-	进程通信：`signal`

###	注意事项

-	子进程报错直接死亡，错误信息默认不会输出到任何地方，所以
	子进程中，多使用`try catch`

##	`os`模块

###	派生进程

```python
os.startfile(filename)
	# 调用系统默认程序打开文件，类似于鼠标双击
	# linux下没有实现
int = os.system(command(str))
	# 运行shell命令，返回命令执行退出状态
	# 和普通shell命令一样默认阻塞，除非后台运算符`&`
```

####	`os.popen`

```python
pipe = os.popen(cmd(str), mode="r"/"w", buffering=-1)
	# 运行shell命令并重定向其输出、输入流到管道开放
```

-	用途：运行shell命令并重定向其输出、输入流到管道开放

-	返回值：返回管道流对象
	-	类似shell中管道语法
	-	管道流对象类似普通文件流，支持一般读、写、迭代
		（但是文档中只有`close`方法）

		```python
		# `r`模式执行，读取子进程标准输出
		pipe.readlines()
		pipe.readline()
		pipe.read()
		for i in pipe:
			# 迭代器语法
		# `w`模式执行，写入子进程标准输入
		pipe.write()
		pipe.close()
			# 返回退出状态
			# 历史原因，退出状态为0时返回None
		```

-	`os.popen`一般不会阻塞，但是python执行需要整个命令行程序
	完成的语句时仍然会阻塞
	-	关闭管道对象
	-	一次性读取所有输出流

> - `subprocess`模块可以实现与`os.system`、`os.popen`相同的
	效果，使用更复杂，但是对流的连接、使用提供更完善的控制

####	`os.fork`

进程分支是构建平行任务的传统做法，是Unix工具集的基本组成部分

-	分支是开始独立程序的直接做法，无论调用程序是否相同
-	分支想法基于复制
	-	程序调用分支例行程序时，操作系统会创建在该进程副本
	-	和进程并行的运行副本
	-	有些系统实现并不真的复制原有程序，以减少资源消耗，
		但是新副本会像真实副本一样运行

```c
import os

def child():
	print("hello from child", os.getpid())
	os._exit(0)

def parent():
	while True:
		newpid = os.fork()
			# 子进程返回0值
			# 父进程返回子进程进程号
		if newpid == 0:
			child()
		else:
			print("hello from parent", os.getpid(), newpid)
		if input() == "q":
			break
```

-	返回值：据此区分父、子进程，执行不同任务
	-	子进程中返回0
	-	父进程中返回子进程ID

-	`os.fork`仅仅是系统代码库中**标准进程分支调用**简单封装
	-	和C共用代码库
	-	在win下标准python版本中无法运行，和win模型冲突过多
	-	Cygwin中python可以运行，虽然行为同真正Unix分支不完全
		相同，但是足够接近

-	`os.fork`实际复制整个python解释器
	-	除非python脚本被编译为二进制机器码

####	`os.exec`

```python
def execv(path, args=tuple/list)
def execl(path, *args=*list/*tuple)
	# 以参数执行指定可执行文件，**代替**当前进程

def execvp(file, args)
def execlp(file, *args)
	# `p` for `$PATH`
	# 在系统搜索路径`$PATH`中定位可执行程序

def execve(path, args=tuple/list, env=dict)
def execle(path, args=*tuple/*list, env=dict)
	# `e` for `environ`
	# 将`env`字典作为环境环境变量传递

def execvpe(path, args=tuple/list, env=dict)
def execlpe(path, args=*tuple/*list, env=dict)
	# 在系统搜索路径`$PATH`中定位可执行程序
	# 将`env`字典作为环境环境变量传递
```

-	`os.exec`族的函数会覆盖当前进程

	-	所以该语句的代码都不会执行
	-	不会更改进程号

	```c
	import os
	param = 0
	while True:
		param += 1
		newpid = os.fork()
		if newpid == 0:
			os.execlp("python", "child.py", str(param))
			assert False, "starting new program error"
				# 上句正常执行，这句永远无法被调用
		else:
			print("child is", pid)
			if input() == "q": break
	```

> - `multiprocessing`模块的进程派生模型+`os.exec`配合使用
	可以在win下实现类似`os.fork`的效果

####	`spawn`

```python
os.spawn
	# 启动带有底层控制的新程序
```
###	进程通信

```python
read_fd, write_fd = os.pipe()
	# 创建管道，返回管道写入、读取描述符
	# 可以使用`os.fdopen()`封装管道描述符，方便读、写
os.mkfifo(path, mode=438, *, dir_fd=None)
	# 创建命名管道，win不支持
	# 仅创建外部文件，使用需要像普通文件一样打开、处理
```

##	`subprocess`模块

###	`Popen`

```python
class Popen:
	def __init__(self,
		args(str,[str]),
		bufsize=-1/0/1/int,
		executable=None/str,
		stdin=None/stream,
		stdout=None/stream,
		stderr=None/stream,
		preexec_fn=None/callable,
		close_fd=obj,
		shell=False/True,
		cwd=None/path,
		env=None/dict,
		universal_newlines=False/True,
		startupinfo=None,
		creationflags=0,
		restore_signals=True/False,
		start_new_session=False/True,
		pass_fds=(),
		encoding=None/str,
		errors=None
	)
```

-	用途

-	参数
	-	`args`：需要执行的命令

	-	`executable`：备选执行命令

	-	`stdin`/`stdout`/`stderr`：执行程序标准输入、输出、
		错误流连接对象
		-	默认：当前进程标准输入、输出、错误
		-	`subprocess.PIPE=-1`：当前管道对象标准...

	-	`preexec_fn`：子进程执行前在子进程中调用的对象
		-	*POSIX* only

	-	`close_fds`：控制关闭、继承文件描述符

	-	`shell`：是否通过shell执行命令
		-	执行shell内置命令则必须置`True`
			-	win：`type`
			-	linux：`set`
		-	Linux下`False`：由`os.execvp`运行

	-	`cwd`：子进程执行目录

	-	`env`：子进程环境变量

	-	`universal_newlines`：是否在3个标准流中使用行结尾符
		-	即是否按照文本处理3个标准流

	-	`startupinfo`：windows only
	-	`restore_signals`：POSIX only
	-	`start_new_session`：POSIX only
	-	`pass_fds`：POSIX only

####	`.communicate`

```python
(stdout, stderr) = communicate(self, input=None, timeout=None)
```

-	用途：和子进程交互，阻塞直到子进程终止
	-	`input`传递给子进程标准输入
	-	返回标准输出、错误输出
	-	`universal_newlines`为`True`时，`input`参数、返回值
		应该为`str`，否则为`bytes`

####	其他方法

```python
def kill(self):
	# kill进程通过*SIGKILL*信号

def terminate(self):
	# 终止进程通过*ISGTERM*信号

def send_signal(self, sig):
	# 向进程传递信号

def poll(self):
	# 检查子进程是否终止，设置、返回`.returncode`属性

def wait(self, timeout=None, endtime=None):
	# 等待直到子进程终止，返回`.returncode`属性

pipe.stdin
pipe.stdout
pipe.stderr
	# 管道标准输入、输出、错误流
	# 创建子进程时，可以选择和子进程相应流连接
	# 支持读、写、迭代
pipe.returncode
	# 子进程退出状态
```

-	`Popen`创建对象对象之后会立刻执行

-	同时指定`stdout`、`stdin`参数可以实现管道双工工作
	-	需要注意，读写时交替发送缓冲数据流可能导致死锁

###	`call`

```python
subprocess.call(
	"type hello.py",
	shell=True/False
)
```

##	`_thread`

-	为系统平台上的各种线程系统提供了可移植的接口
-	在安装了*pthreads POSIX*线程功能的系统上，接口工作方式
	一致，无需修改源码即可正常工作

> - 基本可以完全被`threading`模块替代了

###	`start_new_thread`

```c
def start_new_thread(
	callable,
	args=tuple/list,
	kwargs=dict
)
def start_new():
	# deprecated，同上
```

-	用途：开启新线程，以参数调用`callable`

-	返回值：应该是线程起始地址

-	派生线程在函数返回后退出

	-	若在线程中函数抛出未捕捉异常，打印堆栈跟踪记录、退出
		线程
	-	程序其他部分继续运行 

-	大多数系统平台上，整个程序主线程退出时，子线程随之退出
	-	需要一些处理避免子线程意外退出

###	其他方法

```python
Lock = _thread.alloacate_lock()
	# 获取一个`Lock`锁
	# 等价于`threading.Lock()`
Lock = _thread.allocate()
	# deprecated，同上

RLock = _thread.RLock()
	# 获取一个`RLock`可重入锁
	# 等价于`threading.RLock()`

def _thread.exit()
	# 退出当前线程，可捕捉
	# 等价于显式`raise SystemExit`、`sys.exit()`
def _thread.exit_thread()
	# 同上
```

###	例子

####	例1

-	全局线程锁保护对输出流写入
-	全局线程锁实现主线程、子线程间通信，保证主线程在子线程
	之后退出

```python
import _thread as thread

stdoutmutex = thread.allocate_lock()
	# 创建全局标准输出锁锁对象
exitmutexes = [thread.allocate_lock() for _ in range(10)]
	# 为每个线程创建锁
exitmutexes_bool = [False] * 10
	# 线程共享内存，同样可以使用全局变量作为信号量，而不用
		# 额外开销

def counter(myId, count):
	for i in range(count):

		stdoutmutex.acquire()
			# 线程向标准输出流写入时，获得锁
		print("[%s] => %s" % (myId, i))
		stdoutmutex.release()
			# 向标准输出流写入完毕后，释放锁

		with stdoutmutex:
			# 线程锁同样支持`with`上下文管理器
			print("[%s] => %s again" % (myId, i))

	exitmutexes[myID].acquire()
		# 线程执行完毕后获取对应自身id的锁，通知主线程

for i in range(10):
	thread.start_new_thread(counter, (i, 100))
	# 创建、启动新线程

for mutex in existmutexes:
	# 检查所有信号锁
	while not mutex.locked():
		# 直到信号锁被获取，结束死循环
		pass
print("main thread exiting")
```

####	例2

-	`with`上下文管理器使用锁
-	全局变量实现主线程、子线程通信，避免主线程在子线程之前
	退出

```python
import _thread as thread
import time

stdoutmutex = thread.allocate_lock()
exitmutexes_bool = [False] * 10

def counter(myId, count):
	for i in range(count):
		with stdoutmutex:
			# 线程锁同样支持`with`上下文管理器
			print("[%s] => %s again" % (myId, i))
	exitmutexes[myID] = True

for i in range(10):
	thread.start_new_thread(counter, (i, 100))

while not all(exitmutexes):
	time.sleep(0.25)
	# 暂停主线程，减少占用CPU进行无价值循环
print("main thread exiting")
```

##	`threading`

###	`Thread`

```python
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
```

-	用途：可控线程类，有两种方法使用

	-	传递callable参数创建新对象
	-	继承、覆盖`run`方法：代码和`Thread`深耦合，可能
		不方便代码复用，如`multiprocessing`模块

-	参数

	-	`group`：保留参数用于未来扩展
	-	`target`：可执行对象，将被`run` invoke
	-	`name`：线程名，缺省`Thread-N`
	-	`args`：用于invoke `target`参数tuple
	-	`kwargs`：用于invoke `target` keyword参数dict
	-	`daemon`：是否是守护线程
		-	默认情况下，主进程（线程）会等待子进程、线程退出
			后退出
		-	主进程（线程）不等待守护进程、线程退出后再退出
		-	注意：主进程退出之前，守护进程、线程会自动终止

> - 若衍生类覆盖此构造器方法，务必首先调用此方法

####	`.run`

```python
def run(self):
	pass
```

-	用途：代表线程活动
	-	原`run`用于invoke `target`
	-	覆盖此方法设置线程活动

####	`.start`

```python
def start(self):
	pass
```

-	用途：开始线程活动
	-	线程创建完成后不会立即执行，需要手动调用`.start`启动
	-	多次调用`raise RuntimeError`

####	`.join`

```python
def join(self,
	timeout=None/float):
	pass
```

-	用途：等待直到线程结束
	-	*join*：将线程加入当前线程
	-	可以多次调用
	-	试图导致死锁时，将会`raise RuntimeError`

-	参数
	-	`timeout`：指定超时时间，单位秒
		-	缺省否则阻塞直到线程结束

####	其他方法

```python
bool = is_alive(self):
	# 返回线程是否存活

def setDaemon(self, daemonic):
	# 设置守护进程
bool = isDaemon(self):

def setName(self, name):
	# 设置线程名
def getName(self):
```

###	`Event`

```python
class Event():
	def set(self):
		# 设置信标值为`True`，发送信号

	bool = is_set(self):
		# 查看信标是否被设置

	bool = wait(self, timeout):
		# 阻塞，直到超时、信标被设置为`True`
		# 返回信标值，即因超时返回时返回`False`

	def clear():
		# 重置Event对象，设置信标值为`False`

```

-	用途
	-	发送信号：`is_set`触发事件
	-	接收信号：`wait`阻塞直到事件发生

-	`Event`中包含**信标**，可在线程中设置、接收，实现线程
		间同步

	-	`Event`对象信标默认设置为`False`，等待Event对象线程
		会阻塞直到信标设置为真

	-	若有线程设置信标为真，则将唤醒**所有**等待该`Event`
		对象线程

> - 若只想**唤醒单个线程**，用信号量、`Condition`代替

####	`.clear`

`.clear`可以重置`Event`对象

-	难以确保安全清理、重新赋值`Event`对象，可能导致错过事件
	、死锁

-	且无法保证重置`Event`对象的代码能在线程再次等待此`Event`
	信号之前执行

-	所以`Event`对象最好单次使用，即其信标设置为真应立刻丢弃

> - 若线程需不停重复使用`Event`对象，使用`Condition`代替

###	`Condition`

```python
class Condition():
	def __init__(self, lock=None)
		# `lock`：`Lock`、`RLock`对象，被用作底层锁
		# 缺省创建新`RLock`对象作为底层锁

	bool accquire():
		# 获取一次condition内部锁

	bool release():
		# 释放一次condition内部锁

	def notify(self, n=1):
		# 唤醒至多`n`个on this condition的线程

	def notify_all(self):
		# 唤醒所有on this condition的线程

	bool = wait(self, timeout=None):
		# 释放底层锁，阻塞
		# 直到被唤醒再次获取底层锁、超时，返回

	bool = wait_for(self, predicate(callable), timeout=None):
		# `wait`直到`predicate`返回`True`
```

-	用途：`Condition`对象`wait`等待信号、`notify`唤醒一定
	数量线程实现线程同步

-	说明
	-	以上所有方法执行前均需要已获得底层锁，否则
		`raise RuntimeError`
	-	因此以上方法一般都需要放在`with`代码块中，保证已经
		获取了内部锁

####	`with`上下文管理器

```python
with c:
	c.wait()

with c:
	c.notify()
```

-	`with`进入：获取condition底层锁，保证调用方法前已经获得
	底层锁
-	`with`退出：释放condition底层锁

> - `Condition`支持`with`上下文管理器，而且非常**必须**，
> - 在`help`里面看不到`.acquire`、`.release`方法，但是是有
	而且可以调用的，应该是官方不建议使用

####	`.wait`

-	用途
	-	方法**先释放底层锁**，阻塞，使得其他等待
		**获取此对象底层锁**获得锁
	-	等待被`notify`唤醒，再次获取锁，继续执行

-	底层锁是`RLock`时

	-	`.wait`不是调用其`.release()`方法，而是调用`RLock`
		内部方法确保真正释放锁，即使`RLock`被递归的获取多次

	-	再次获取锁时，调用另一个内部接口恢复递归层次，即
		`RLock`内部计数

	-	`RLock`本身性质：在没有被持有时，其内部计数被重置
		为1，其他线程可以自由获取

####	`.notify`

-	`.notify`并不释放condition底层锁
-	只是控制能够获取底层锁的线程数量

###	`Semaphore`

```python
class Semaphore(builtins.object):
	def __init__(self, value):
		# `value`：起始许可证数量（最多允许同时执行线程数目）

	bool = acquire(self, blocking=True, timeout=None):
		# 获取信号量，内部计数（许可证）减1

	def release():
		# 释放信号量，内部计数（许可证）加1
```

-	用途：`Semaphore`对象`release`方法生产、`acquire`方法
	消耗信号量，实现线程通信

	-	可以像标准锁一样使用信号量作线程同步，但会增加复杂性
		影响性能
	-	更适用于需要在线程间引入信号、限制的程序，如限制代码
		的并发访问量

> - 信号量对象是建立在共享计数器基础上的同步原语

####	`.acquire`

-	用途：获取信号量，内部计数（许可证）大于0则立刻减1

	-	内部计数`>0`，`-1`立即返回`True`
	-	内部计数`=0`，阻塞、等待，直到其他线程调用`release`

-	返回值：成功获取许可证则返回`True`，否则返回`False`

####	`.release`

-	用途：释放信号量，内部计数（许可证）加1
	-	内部计数`=0`，表明有线程阻塞，**随机**唤醒线程

###	`BoundedSemaphore`

```python
class BoundedSemaphore(Semaphore):
	def release(self):
		# 释放信号量，内部计数加1
		# 当信号量总数超过初始化值时`raise ValueError`
```

###	`Lock`

> - `Threading.Lock`等价于`_thread.allocate_lock`，二者
	都是工厂方法，返回`lock`类的实例

```python
class lock:
	bool = acquire(blocking=True/False, timeout=-1)
		# 尝试获得锁，返回是否获得锁
	bool = acquire_lock
		# deprecated，同`acquire`

	bool = locked()
		# 返回锁状态
	bool = lockec_lock()
		# deprecated，同`lock`

	bool = release()
		# 释放锁，若锁本身未获取`raise RuntimeError`
	bool = release_lock()
		# deprected，同`release`
```

-	用途：同步原语，排他性使用某资源

-	说明
	-	支持`with`上下文语法，代码块执行前自动获取锁，执行
		结束后自动释放
	-	为了避免出现死锁，每个线程应该一次只允许获取一个锁，
		否则应该使用更高级死锁避免机制
	-	适合简单的锁定可变对象

> - `lock`对象应该是C实现，里面的方法是没有`self`参数的

####	其他

`Lock`锁可以视为**使用更加方便的全局变量**

-	可以用于线程之间的通信：给每个子线程分配单独一个锁，
	主线程、子线程可以通过锁状态通信

-	很大程度上可以被全局变量“替换”

	-	获得锁：不断检查全局变量状态，阻塞直到全局变量
		状态代表锁可获得，修改全局变量状态代表锁被获取

	-	释放锁：修改全局变量状态代表锁可获得

	> - 不断检查变量状态会无意义占用CPU时间，可以在检查
		间隙使用`time.sleep()`暂停线程

###	`RLock`

> - 工厂方法，返回`RLock`实例

```python
class RLock:
	bool =  acquire(block=True):
		# 尝试获取锁，返回是否获取锁
		# 每次获取锁，内部计数器加1
	bool = release()
		# 释放锁，若锁本身未获取`raise RuntimeError`
		# 每次释放锁，内部计数器减1
```

-	用途：可重入锁，可以被同一线程多次获取
	-	若锁被当前线程获取多次，则需要被释放同样次数才能被
		其他线程获取
	-	即只有内部计数器回到初始状态才能被任意线程获取

-	没有线程持有锁时，`RLock`内部计数被**重置为1**

	-	应该是`RLock`就是通过内部计数记录被获取次数

-	常用于**给类整体上锁**
	-	类内部每个方法都获取锁
	-	类内部方法之间相互调用
	-	这样类实例方法每次只能有一个线程完整调用

	```python
	import Threading

	class SharedCounter:
		_lock = threading.RLock()
			# 被所有实例共享的类级锁
			# 需要大量使用计数器时，内存效率更高
		def __init__(self, intial_value=0):
			self._value = initial_value

		def incr(self, delta=1):
			with ShareCounter:
				self._value += delta

		def decr(self, deta=1):
			with SharedCounter:
				# 获取锁之后，调用也需要获取锁的`.incr`方法
				self.incr(-delta)
	```

###	`local`

-	用途：创建本地线程存储对象，该对象属性保存、读取操作只对
	当前线程可见
	-	可以用于保存当前运行线程状态，**隔离**不同线程间数据
		-	锁
		-	套接字对象

###	例子

####	提前终止线程

#####	轮询方法

-	将**函数封装在类**中，在类中**设置成员变量作为轮询点**
-	方法`terminate`改变轮询点状态，用于外部调用结束线程
-	线程方法`run`检查轮询点状态判断是否结束自身
	-	线程执行类似IO的阻塞操作时，无法返回自身、无法检查
		轮询点，通过轮询终止线程难以协调


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
	t = Thread(karget=c.run, args=(10,0))
		# 创建Thread
	t.start()
		# 启动线程
	c.terminate()
	c.join()
```

#####	超时循环

-	设置任务超时，超时自动返回
	-	任务只要超时就会返回，不会出现线程阻塞

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
			...continued processing...
		...terminated...
		return
```

####	线程同步、通信

#####	`Event`方式

```python
from threading import Event, Threading

def send_signal(start_evt):
	start_evt.set()
		# 设置事件

def recv_signal():
	start_evt = Event()
		# 创建新事件
	t = Thread.threading(target=send_signal, start_evt)
	start_evt.wait()
		# 阻塞直到接收到信号
```

#####	`Condition`实现`queue.Queue`

-	自定义数据类型，封装`Condition`实例进行线程间同步
-	`put`方法生产，`notify`通知消费者
-	`get`方法消费，阻塞直到被唤醒

```python
import heapq
from threading import Condition

class PriortyQueue:
	def __init__(self):
		self._queue = [ ]
		self._count = 0
		self._cv = Condition()
			# 封装`Condition`实例实现线程间同步

	def put(self, item, priority):
		with self._cv:
			heapq.heappush(self._queue, (-priority, self._count, item))
			self._count += 1
			self._cv.notify()

	def get(self):
		with self._cv:
			# 阻塞，直到空闲
			while len(self._queue) == 0:
				self._cv.wait()
				# `Condition`默认使用`RLock`，可重入
			return heapq.heappop(self._queue)[-1]
```

####	防死锁机制

#####	严格升序使用锁

-	`local`保存当前线程状态，隔离不同线程锁数据

```python
from threading import local
from contextlib import contextmanager

_local = local()

@contextmanager
	# 使支持`with`上下文管理器语法
def acquire(*locks):

	locks = sorted(locks, key=lambda x: id(x))
		# 根据*object identifier*对locks排序
		# 之后根据此list请求锁都会按照固定顺序获取

	acquired = getattr(_local, "acquired", [ ])
		# 为每个线程创建独立属性保存锁
	if acquired and max(id(lock)) for lock in acquired >= id(locks[0]):
		# `_local`中已有锁不能比新锁`id`大，否则有顺序问题
		raise RuntimeError("lock order violation")

	acquired.extend(locks)
	_local.acquired = acquired
		# 更新线程环境中锁

	try:
		for lock in locks:
			# 只允许升序获取锁
			lock.acquire()
		yield
			# `with`语句进入处
	finally:
		# `with`语句退出处
		for lock in reversed(locks):
			# 只允许降序释放锁
			lock.release()
		del acquired[-len(locks):]
```

```python
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

##	`queue`模块

###	`Queue`

```python
class Queue(builtins.object):
	def __init__(self, maxsize=0):
		# `maxsize`；限制可以添加到队列中的元素数量
```

`queue.Queue`：创建被多个线程共享的`Queue`对象

-	线程安全的数据交换方式，基于`collections.deque`

	-	`Queue`对象已经包含必要的锁，可以在多个线程间安全的
		共享数据
	-	`Queue`实际上是在线程间传递**对象引用**，不会复制
		数据项，如果担心对象共享状态，可以传递不可修改数据
		结构、深拷贝

> - 还可以使用`Condition`变量包装数据结构，实现线程线程中
	间通信

####	`get`

```python
obj =  get(self,
	block=True,
	timeout=None/num)

obj =  get_nowait(self):
	# 等同于`get(block=False)`
```

-	用途：从队列中移除、返回一个元素

-	参数
	-	`block`
		-	`False`：没有空闲slot立即`raise Empty exception`

####	`.task_done`

```python
def task_done(self):
	# 指示前一个队列“任务”完成
def join(self):
	# 阻塞直到队列中所有元素被获取（消费）、**处理**
```

-	说明
	-	`.join`阻塞时，要所有队列中元素都被告知`task_done`，
		才能解除阻塞
	-	即队列消费者每次`get`元素、处理后，要**手动**调用
		`task_done`告知队列任务已完成

> - 也可以将`Event`和队列元素封装，用`Event`对象告知队列元素
	处理完成

####	`.put`

```python
def put(self,
	item,
	block=True,
	timeout=None/num):
	pass

def put_nowait(self, item):
	# 等价于`put(self, item, block=False)`
```

-	用途：向队列中追加元素

-	参数
	-	`block`
		-	`False`：没有空闲slot立即`raise Full exception`

####	`.qsize`

```python
int = qsize(self):
	# 返回队列大概大小（不可靠）
	# 非线程安全，在其使用结果时队列状态可能已经改变

bool = empty(self):
	# deprecated，使用`qsize() == 0`替代
	# 和`qsize()`一样非线程安全

bool = full(self):
	# deprecated，使用`qsize() > n`替代，非线程安全
```

###	例子

####	队列元素消费通知

#####	`Event`进队列

```python
from queue import Queue
from threading import Thread, Event

def producer(out_q):
	while running:
		evt = Event()
		out_q.put((data, evt))
			# 将`Event`实例放入队列
		evt.wait()
			# 阻塞直到收到消费者信号

 # A thread that consumes data
def consumer(in_q):
	while True:
		data, evt = in_q.get()
		evt.set()
			# 告知生产者消费完成
```

####	协调生产、消费线程终止

#####	队列中添加特殊值

```python
_sentinel = object()

def producer(out_q):
	while running:
		out_q.put(data)
	out_q.put(_sentinel)
		# 将特殊值放入队列，结束生产、通知消费者
def consumer(in_q):
	while True:
		data = in_q.get()
		if data is _sentinel:
			# 消费者接收到特殊值，结束生产
			in_q.put(_sentinel)
				# 特殊信号放回队列，继续传递
			break
```

####	`Queue`实现线程池

```python
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread
from queue import Queue

def echo_client(q):
	sock, client_addr = q.get()
	while True:
		msg = sock.recv(65536)
		if not msg:
			break
		sock.sendall(msg)
	print("closed")

def echo_server(addr, nworkers):
	q = Queue()
	for n in range(nworkers):
		t = Thread(target=echo_client, args=(q,))
		t.daemon = True
		t.start()
	socket = socket(AF_INET, SOCK_STREAM)
	socket.bind(addr)
	sock.listen(5)
	while True:
		client_sock, client_addr = sock.accept()
		q.put((client_sock, client_addr))
```

##	`multiprocessing`模块

-	`multiprocessing`支持一个基本与平台无关的进程派生模型，
	在Unix、Windows下均可正常工作

-	实现方式
	-	启动一个新的Python进程，`import`当前模块
	-	`pickle`序列化需要调用的函数，传递至新进程执行

> - 其中`Pool`、`Queue`、`Pipe`等实际都是其封装其子模块类
	的工厂方法

###	`Process`

```python
class Process():
	def __init__(self,
		group=None,
		target=None/callable,
		name=None/str,
		args=()/list,
		kwargs={}/dict,
		daemon=None
	):
		self.authkey
		self.daemon
		self.exitcode
		self.name
		self.pid

	def is_alive(self):

	def join(self,
		timeout=None/num
	):

	def start(self):
		# 启动进程

	def run(self):
		# `start`方法调用`run`方法，若进程实例未传入`target`，
		# `start`默认执行`run`方法

	def terminate():
		# 立即结束进程
```

-	用途：`Multiprocessing`核心，类似于`Thread`，实现多进程
	创建、启动、关闭

-	成员方法基本类似`Thread`

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

###	`Pool`

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
-	用途：创建管理进程池，提供指定数量进程供用户调用
	-	新请求提交到pool中时
		-	若进程池没有满，将创建新进程执行请求
		-	否则请求等待直到池中有进程结束，然后创建新进程
	-	适合子进程多且需要控制子进程数量时

####	`apply_async`

```python
def apply(self, func, args=(), kwds={}):
	# 分配进程执行`func(*args, **kwds)`，阻塞
	# 返回值：`func(item)`返回值

def apply_async(self, func, args=(), kwds={},
	callback=None, error_callback=None)
	# 异步分配进程执行调用，非阻塞
	# 返回值：`ApplyResult`对象
```

-	返回值：异步非阻塞调用返回结果操作句柄`ApplyResult`
-	回调函数要有返回值，否则`ApplyResult.ready()=False`，
	回调函数永远无法完成

#####	`ApplyResult`

```python
class ApplyResult:
	def __init__(self, cache, chunksize, length,
		callback, error_callback):

	def get(self, timeout=None):

	bool = ready(self):

	bool = successful(self):

	bool = wait(self, timeout=None):
```

####	`map_async`

```python
def map(self, func, iterable, chunksize=None):
	# 同步多进程`map(func, iterable)`，阻塞直到全部完成
	# 返回值：结果列表，结果按调用顺序

def map_async(self, func, iterable, chunksize=None,
	callback=None, error_callback=None):
	# 异步多进程`map`，非阻塞
	# 返回值：`MapResult(ApplyResult)`对象

def imap(self, func, iterable, chunksize=1):
	# 迭代器版`map`，更慢、耗内存更少
def imap_unordered(self, func, iterable, chunksize=1):
	# 返回结果无序版`imap`

def starmap(self, func, iterable, chunksize=1):
	# 同步`map`，参数被解构`func(*item)`，阻塞

def starmap_async(self, func, iterable, chuncksize=None,
	callback=None, error_callback=None):
	# 异步`startmap`，阻塞
```

####	终止

```python
def close(self):
	# 关闭进程池，不允许添加新进程

def join(self):
	# 阻塞，直至进程池中所有进程执行完毕
	# 必须先`.close`进程池

def terminate(self):
	# 终止进程池
```

###	`Queue`

```python
class SimpleQueue():
	bool empty(self):
	def get(self):
	def put(self):
class Queue:
	def __init__(self, maxsize=0, *, ctx):
	def join_thread(self):
		# join feeder线程
	def cancel_join_thread(self):
		# 控制在进程退出时，不自动join feeder线程
		# 有可能导致部分数据在feeder中未被写入pipe而丢失
	def close(self):
		# 关闭feeder线程
	def qsize(self):
	def empty(self):
	def full(self):
	def get(self, block=True, timeout=None):
	def get_nowait(self):
	def put(self, obj, block=True, timeout=None):
	def put(self):
class JoinableQueue(Queue):
	def task_done():
	def join():
```

-	用途：进程安全队列

-	`multiprocessing.Queue`基于`multiprocessing.Pipe`构建

	-	数据传递时不是直接写入`Pipe`，而是写入本地buffer，
		通过feeder线程写入底层`Pipe`，从而实现**超时控制**、
		非阻塞`put/get`

	-	所以提供了`.join_thread`、`cancel_join_thread`、
		`close`函数控制feeder流行为

	-	相当于线程安全的`queue.Queue`的多进程克隆版

-	`multiprocessing.SimpleQueue`：简化队列

	-	没有`Queue`中的buffer，没有使用`Queue`可能有的问题，
		但是`put/get`方法都是阻塞的、没有超时控制

###	`Pipe`

> - `mltiprocessing.Pipe()`返回两个用管道相连的、读写双工
	`Connection`对象

```python
class Connection():
	def __init__(self, handle,
		readable=True,
		writable=True):

	def close(self):
	def fileno(self):
		# 返回描述符或连接处理器
	bool = poll(self, timeout=0):
		# 是否有输入可以读取
	obj =  recv(self):
		# 接收pickable对象
		# 若接收字节流不能被pickle解析则报错
	bytes = recv_bytes(self, maxlength=None):
		# 接收字节流作为`bytes`
	int = recv_bytes_into(self, buf, offset=0):
		# 接收字节流存入可写`buf`
		# 返回读取的字节数量
	def send(self, obj):
		# 发送对象
	def send_bytes(self, buf, offset=0, size=None):
		# 发送bytes-like对象中字节数据
```

-	对象会在每条信息前添加标志字节串，可以自动处理多条信息
	堆叠

	-	可以通过`os.read(connect.fileno())`获取

###	共享内存

```python
class SynchronizedBase:
	def __init__(self, obj, lock=None, ctx=None):

	def get_lock(self):
		# 获取`multiprocessing.synchronize.RLock`

	def get_obj(self):
		# 获取数据对象，C数据结构`ctypes.`
```

####	`Value`

```python
def Value(typecode_or_type, *args, lock=True):
	# 返回同步共享对象
class Synchronized(SynchronizedBase):
	value
```

####	`Array`

```python
def Array(typecode_or_type, size_or_initializer, *,
	lock=True)

def SynchronizedArray(SynchronizedBase):
	def __getitem__(self, i):
	def __getslice__(self, start, stop):
	def __len__(self):
	def __setitem(self, i, value):
	def __setslice__(self, start, stop, values):
```

###	`Manager`

常与`Pool`模块一起使用，共享资源，不能使用`Queue`、`Array`

###	其他方法

```python
import multiprocessing as mltp
l1 = mltp.Lock()
	# 获取进程锁
rl1 = mltp.RLock()
	# 获取可重入锁
s1 = mltp.Semaphore(value=int)
	# 获取信号量对象
bs1 = mltp.BoundedSemaphore(value=int)
	# 获取有上限信号量对象
e1 = mltp.Event()
	# 获取事件对象
cv1 = mltp.Condition(lock=None)
	# 获取Condition对象
cp = mltp.current_process()
	# 获取当前Process对象
```

##	`concurrent.futures`

-	异步并发模块

###	`ThreadPoolExecutor`

```python
class ThreadPoolExecutor:
	def __init__(self,
		max_workers=None,
		thread_name_prefix=''):

	def shutdown(self, wait=True):
		# 清理和该执行器相关资源

	def submit(self, fn(callable), *args, **kwargs):
		# 以指定参数执行`fn`
		# 返回：代表调用的future，可以用于获取结果

	def map(self, fn, *iterables,
		timeout=None,
		chunksize=1/int)
		# 并行`map(fn, iterables)`
		# 返回：按调用顺序的结果迭代器
```

-	用途：创建线程池

####	`.shutdown`

-	用途：关闭执行器，清理和该执行器相关的资源
	-	可以多次调用，调用之后不能进行其他操作

-	参数
	-	`wait`：阻塞，直到所有运行*futures*执行完毕，所有
		资源被释放

###	`ProcessPoolExecutor`

```python
class ProcessPoolExecutor:
	def __init__(self,
		max_workers=None):

	def shutdown(self, wait=True):
		# 清理和该执行器相关资源

	def submit(self, fn(callable), *args, **kwargs):
		# 以指定参数执行`fn`
		# 返回：代表调用的`Future`实例，可以用于后续处理

	def map(self, fn,
		*iterables,
		timeout=None,
		chunksize=1/int)
		# 并行`map(fn, iterables)`
		# 返回：按调用顺序的结果迭代器
```

-	用途：创建进程池
	> - 参考`ThreadPoolExecutor`，`.map`方法支持`chunksize`
		参数

-	常使用`with`语句使用

	```python
	with ProcessPoolExecutor() as Pool:
	```

	-	处理池执行完`with`语句块中后，处理池被关闭
	-	程序会一直等待直到所有提交工作被完成

####	注意事项

-	被提交的任务必须是**简单函数形式**，不支持方法、闭包和
	其他类型可执行

-	函数参数、返回值必须兼容`pickle`模块，因为进程间通信
	交换数据使用其序列化

-	函数不要修改环境
	-	被提交的任务函数出打印日志之类等简单行为外，不应该
		有保留状态、副作用

-	混合使用进程池、多线程时
	-	创建任何线程之前先创建、激活进程池
	-	然后线程使用同样的进程池进行计算密集型工作

###	`Future`

```python
class Future:
	def add_done_callback(self, fn):
		# 添加future完成的回调函数
		# 多次调用添加的回调函数按添加顺序执行

	bool =  cancel(self):
		# 尝试取消当前future，返回是否成功取消
		# 正在运行、执行完成的future不能被取消

	bool = cancelled(self):
		# 查看当前future是否被取消

	bool = done(self):
		# 查看当前future是否被取消、完成

	def exception(self, timeout=None):
		# 返回当前future代表的调用的exception

	def result(self, timeout=None):
		# 返回当前future代表调用的返回值
```

-	用途：代表一个异步计算的结果
	-	直接创建对象无价值

##	`socket`模块



