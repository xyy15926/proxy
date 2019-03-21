#	Python Iterator

##	迭代器

可迭代对象/迭代器

-	实现`__next__`方法
-	通过`yield`迭代值

###	`next`

一般可以使用`for`循环遍历可迭代对象，也可以使用`next`方法
手动遍历可迭代对象

```python
Val = next(iterator[, default])
	# 返回迭代器下个元素
	# 迭代器消耗完毕，返回`default`而不`raise StopIteration`
```

-	`next`函数调用可迭代对象的`__next__`方法获取下个迭代对象
-	`StopIteration`异常标识迭代完成
	-	使用`next`手动迭代，需/可在代码中捕获、处理，`next`
		也可以使用指定值标识结尾
	-	`for`语句可以自动处理

```python
def manual_iter():
	with open("/etc/passwd") as f:
		try:
			while True:
				line = next(f)
					# `next`手动遍历获取下个迭代元素
				print(line, end ="")
		except StopIteration:
			# 捕获`StopIteration`，指示达到结尾
			pass
```

###	`yield`

`yield`语句将函数转换为**生成器（generator）**

-	相较于普通函数
	-	函数调用并不执行实际代码，返回值为iterable对象
		（iterator）
	-	若需要得到具体结果，需要遍历（迭代`next`）iterator
	-	生成器函数返回、退出，即iterator迭代终止并
		`raise StopIteration`
-	比起类的实例
	-	无需设置属性保存状态计算下个`next`值
	-	代码简洁、执行流程清晰
-	对于返回的iterator
	-	`next(iterator)`则代码执行至`yield`停止，`yield`值
		作为iterator元素
	-	继续`next(iterator)`则从`yield`下条语句开始执行，
		直到再次遇到`yield`语句（一般是在循环语句中）
-	`isgeneratorfunction`可以判断函数是否为generator

```python
def frange(start, stop, increment):
	x = start
	while x < stop:
		yield x
			# `yield`语句将函数转换为生成器
			# 生成器只能用于迭代操作
		x += increment

def fab(max=0):
	n, a, b = 0, 0, 1
	while max == 0 or n < max:
		yield b
		a, b = b, a + b
		n = n + 1
```

###	`iter`

`iter`函数调用“可迭代”对象的`__iter__`方法

####	代理迭代

-	`__iter__`方法**应该**返回迭代器对象
-	可以自定义对象的`__iter__`方法，返回与当前对象“不相关”
	的可迭代对象，实现“代理”

```python
class Node:
	def __init__(self, value):
		self._value = value
		self._children = [ ]

	def __repr__(self):
		return "Node({!r})".format(self._value)

	def add_children(self, node):
		self._children.append(node)

	def __iter__(self):
		return iter(self._children)
		# 将迭代请求传递给内部`__children`属性
		# `iter`函数简单的通过调用`__iter__`方法返回迭代器
			# 对象
		# python迭代器协议需要`__iter__`返回了实现`__next__`
			# 方法的迭代器对象

	def depth_first(self):
		yield self
			# 先序深度优先遍历
		for c in self:
			# 处理子节点
			yield from c.depth_first()

def test():
	root = Node(0)
	child1 = Node(1)
	child2 = Node(2)
	root.add_child(child1)
	root.add_child(child2)
	for ch in root:
		print(ch)

def test_2():
	root = Node(0)
	child1 = Node(1)
	child2 = Node(2)
	root.add_child(child1)
	root.add_child(child2)
	child1.add_child(Node(3))
	child1.add_child(Node(4))
	child2.add_child(Node(5))

	for ch in root.depth_first():
		print(ch)
```

####	关联迭代器

>	python迭代协议
> - `__iter__`方法返回一个特殊的迭代器对象
> - 迭代器对象实现了`__next__`方法，并通过`StopIteration`
	异常标识迭代完成

也可以通过使用关联迭代器实现`depth_first`方法，但如此必须在
迭代处理过程中维护大量的出状态信息，不如将迭代器定义为生成器
方便

```python
class Node2:
	def __init__(self, value)"
		self._value = value
		self._children = [ ]
	
	def __repr__(self):
		return "Node({!r})".format(self._value)

	def add_child(self, node):
		self._children.append(node)

	def __iter__(self):
		return iter(self._children)

	def depth_first(self):
		return DepthFirstIterator(self)

class DepthFirstIterator(object):

	def __init__(self, start_node):
		self._node = start_node
			# 根节点
		self._children_iter = None
			# 当前遍历节点**兄弟节点**迭代器
		self._child_iter = None
			# 当前遍历节点**子节点**迭代器

	def __iter__(self):
		return self

	def __next__():
		if self._children_iter is None:
			self._children_iter = iter(self._node)
			return self._node
		elif self._child_iter:
			try:
				nextchild = next(self._child_iter)
				return nextchild
					# 从这里看，这个迭代器最多只支持3层树
			except StopIteration:
				self._child_iter = None
				return next(self)
					# 递归调用`__next__`
		else:
			self._child_iter = next(self._children_iter).depth_first()
			return next(self)
				# 递归调用`__next___`
```

####	带有外部状态的生成器函数

###	`reversed`

使用内置`reversed`函数，可以反向迭代一个序列

-	`reversed`函数调用对象的`__reversed__`方法，需对象预先
	实现`__reversed__`方法
-	将对象转换为一个列表（已实现`__reversed__`方法的类型），
	然后也可以应用`reversed`，但这样会耗费大量资源

```python
class Countdown:
	def __init__(self, star):
		self.start = start

	# forward iterator
	def __iter__(self):
		n = self.start
		while n > 0:
			yield n
			n -= 1

	# reverse iterator
	def __reversed__(self):
		n = 1
		while n <= self.start:
			yield n
			n += 1

def test():
	for rr in reversed(Countdown(30)):
		print(rr)
	for rr in Countdown(30):
		print(rr)
```

###	带有外部状态生成器

如果需要将生成器的某**内部状态（属性）**暴露给用户，可以将
其实现为一个类

-	生成器函数放在`__iter__`方法中
-	生成器类可以当作普通的生成器函数使用
-	公有属性、方法暴露出来，用户可以直接使用
-	生成器需要跟程序其他部分打交道，会导致代码复杂，使用类
	定义生成器的方法，可以将这些逻辑打包，而不改变算法逻辑
-	需要注意的是，如果不是使用`for`语句进行迭代，需要先调用
	`iter`函数获取生成器，这与使用函数生成器用法有差

```python
from collections import deque

class linehistory:
	def __init__ (self, lines, histlen=3):
		self.lines = lines
		self.history = deque(maxlen=histlen)
			# 需要暴露的内部状态

	def __iter__(self):
		for lineno, line in enumerate(self.lines, 1):
			self.history.append((lineno, line))
			yield line

	def clear(self):
		self.history.clear()

def test():
	with open("somefile.txt") as f:
		lines = linehistory(f)
		for line in lines:
			if "python" in line:
				for lineno, hline  in lines.history:
					print("{}:{}".format(lineno, hline), end ="")

	f = open("somefile.txt")
	lines = linehistory(f)
	it = iter(lines)
		# 不同于函数生成器，非`for`迭代场合需要调用`iter`
			# 获取生成器
		# 当然，应该也可以实现`__next__`方法
	print(next(it))
```

##	迭代器常用函数

###	单迭代器

####	`itertools.islice`

迭代器切片

迭代器、生成器不能使用标准切片操作
-	其长度实现不知道
-	没有实现索引
-	`islice`可以返回生成指定元素的迭代器，通过遍历并丢弃直到
	切片开始索引位置元素
	-	`islice`同样会消耗迭代器的数据、不可逆

```python
from itertools import islice

def count(n):
	while True:
		yield n
		n += 1

def test():
	c = count(0)
	for x in itertools.islice(c, 10, 20):
		# 迭代器切片
		print(x)
```

####	`itertools.dropwhile`

跳过开始部分

```python
from itertools import dropwhile

def test():
	with open("/etc/passwd") as f:
		for line in dropwhile(lambda line: lines.startwith("#"), f):
			# 跳过开头元素，直到传入函数返回`False`
			print(line, end="")
```

####	`itertools.permutations`

返回集合中所有元素可能**排列**的迭代器

```python
from itertools import permutations

def test():
	items = ["a", "b", "c"]
	for p in permutations(items, r=2):
		# 接受集合，返回一个元组序列，每个元素由集合中`r`个
			# 元素的一个可能排列组成
		print(p)
```

####	`itertools.combinations`

返回集合中所有元素可能**组合**的迭代器

```python
from itertools import combinations

def test():
	items = ["a", "b", "c", "d"]
	for c in combinations(items, r=3):
		print(c)
```

####	`itertools.combinations_with_replacement`

允许元素**重复**组合迭代器

```python
from itertools import combinations_with_replacement

def test():
	items = ["a", "b", "c", "d"]
	for c in combinations_with_replacement(items, r=3):
		print(c)
```

###	“组合”迭代

####	`enumerate`

加索引值迭代

```python
def test():
	items = ["a", "b", "c"]
	for idx, val in enumerate(items, start=1):
		# 从`start`开始enumerate，返回迭代器
			print(idx, val)

def parse_data(filname):
	with open(filename, "rt") as f:
		for lineno, line in enumerate(f, 1):
			fields = line.split()
			try:
				count = int(fields[1])
				...
			except ValueError as e:
				print("line {}: parse error".format(lineno, e))
```

####	`zip`

同时迭代多个序列

```python
from itertools import zip_longest

def test():
	a = [1, 2, 3]
	b = ["a", "b", "c", "d", "e"]
	c = ["w", "x", "y", "z"]
	for i in zip(a, b, c):
		# 以“短”的迭代器为准，返回迭代器
		print(i)
	for i in zip_longest(a, b, c):
		# 以“长”的迭代器为准，“短”者从头循环
		print(i)
```

####	`itertools.chain`

连接迭代器
-	不要求迭代器中元素类型相同
-	节省内存资源

```python
from itertools import chain

def test():
	a = [1, 2, 3, 4]
	b = ["x", "y", "z"]
	for x in chain(a, b):
		print(x)
```

##	迭代器设计模式

###	数据处理管道

使用生成器函数实现管道机制

-	可以解决各类问题：解析、读取实时数据、定时轮询
-	`yield`作为数据生产者，`for`循环语句作为数据消费者，
	生成器被连接在一起后，每个`yield`会将单独的数据元素传递
	给**迭代处理管道**的下一阶段
-	这种模式下，每个生成器函数小、独立，容易编写、维护、重复
	使用、理解
-	避免将大量数据一次性放入内存中，节省内存资源
-	但是如果像立即处理所有数据，管道方式可能不使用，但是可以
	将这类问题从逻辑上变为工作流处理方式

```python
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
	for path, dirlist, filelist in os.walk(top):
		for name in fnmatch.filter(filelist, filepat):
			yield os.path.join(path, name)

def gen_opener(filenames):
	for filename in filenames:
		if filename.endswithc(".gz"):
			f = gzip.open(filename, "rt")
		elif filename.endswith(".bz2"):
			f = bz2.open(filename, "rt")
		else:
			f = open(filename, "rt")
		yeild f

		f.close()
			# 处理完之后需要关闭文件

def gen_concatenate(iterators):
	# 类似于`itertools.chain`，将输入序列拼接为长序列
	# 但`itertools.chain`会一次性消耗所有传入的生成器，而
		# `gen_opener`生成器函数每次打开的文件，在下次迭代
		# 步骤中已经关闭，因此不能使用`chain`
	for it in iterators:
		yeild from it
			# `yield from`将`yield`操作代理到父生成器上

def gen_rep(pattern, lines):
	pat = re.compile(pattern)
	for line in lines:
		if pat.search(line)
			yield line

def test():
	lognames = gen_find("access-log*", "www")
	files = gen_opener(lognames)
	lines = gen_concatenate(files)
	pylines = gen_grep("(?i)python", lines)

	bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
	bytes = (int(x) for x in bytecolumn if x != "-")
	print("Total", sum(bytes))
```

###	展开嵌套序列

```python
from collections import Iterable

def flatten(items, ignore_type=(str, bytes)):
	for x in items:
		if isinstance(x, Iterable) and not isinstance(x, ignore_types):
		# `isinstance(x, ignore_types)`将字符串、字节串排除
			# 其也是iterable，防止其展开成单个字符
			yield from flatten(x)
				# `yield from`可以简化在生成器中调用其他
					# 生成器，否则只能使用`for`循环
		else:
			yield x

def test():
	items = [1, 2, [3, 4, [5, 6], 7,], 8]
	for x in flatten(items):
		print(x)
```

###	排序合并**有序**迭代对象

```python
import headpq

def test():
	a = [1, 4, 7, 10]
	b = [2, 5, 6, 11]
		# 原iterable必须有序（`key(val)`结果从小到大）
		# 这个`merge`就是简单的merge
	for c in heapq.merge(a, b,
		key=lambda x: 10 - x,
		reverse=True):
		# 返回一个迭代器，因此内存资源消耗小
		print(c)
```

###	替代`while`循环

```python
Iterator = iter(iterable)
	# `iterable`必须其自身提供其自己的iterator
Iterator = iter(
	callable,
	sentinel)
	# 创建iterator invoke `callable`直到其返回`sentinel`
	# 返回值作为迭代器元素，不包括`sentinel`
```

使用`iter`的第二种模式可以替代`while`循环

```python
import sys
CHUNKSIZE = 8192

def test():
	f = open("/etc/passwd")
	for chunk in iter(lambda: f.read(CHUNKSIZE), b""):
		n = sys.stdout.write(chunk)
```




