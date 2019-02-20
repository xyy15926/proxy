#	Python一些典型问题

##	编码问题

###	流程

-	python解释器编码：处理
-	python代码文件编码：输入
-	terminal编码：输入、输出
-	操作系统语言设计：显示

###	Python2

python2默认编码方案是ascii（str类型编码方案）
（不更改`sys.defualtencoding`）

-	输入ascii字符集以外的字符时，也认为是ascii编码字节流，
	并采用码元长度（1B）存储
-	采用和ascii编码方案不兼容的其他编码方案存储的文件会因为
	python2无法“理解”报错

####	Python2中`str`类型

python2中`str`类型实际上应该看作是**字节串**，`str`索引是按
字节，有`decode`方法用于解码，将字节流转换为真正的unicode
**字符串** 

###	Python3

python3使用utf-8编码方案，默认将输入字节流视为utf-8编码
字节流

####	Python3中的`str`类型

-	`str`类型的逻辑变成真正的**字符串**，虽然每个字符长度
	可能不同
-	有`encode`方法，将字符串编码为其他编码方案的字节流
	（包括自身的utf-8）

####	字节串输出

但是python3和2中对于**字节串**（2中就是`str`，3中是`bytes`）
字节序列化（**字符串化**）输出处理方法不同

-	python2中应该是将`str`直接作为字节流传给系统

-	python3则将`bytes`视为**ascii编码序列**，**解码**后输出
	（"可打印acsii字符打印，否则使用16进制表示"这样的
	字符串流）

因为输出函数也只是个函数，并不是按照内存“输出”

#	todo
python解释器和terminal编码方案不同测试

##	Module、Package

-	module：模块，`.py`文件
-	package：包，含有`__init__.py`的文件夹

###	导入模块

就导入模块而言，python对modules、packages的处理有些差异，
不能将mudules、packages的导入等同

|操作|`import`|`from`|`self.cld`|
|-----|-----|-----|-----|
|packages|是|是|取决于`__init__.py`|
|modules|是|是|是|
|attrs|否|是|是|
|imported|否|是|同以上中某个|

其中

-	`import`；能否通过`import`语句导入
-	`from`：能否通过`from ... import ...`语句导入
-	`self.cld`：能否使用`self.cld`
	-	对于modules、attrs显然可以，`cld`必然在`dir()`中
	-	对于packages，导入可以看作是导入`__init__.py`单个
		文件，因此取决于`__init__.py`中是否导入**显式**相应
		**后代**modules、attrs
		-	对于外部元素，`from outer import *`即可添加进
			命名空间
		-	而**后代**元素必须**显式**依次导入名称
-	imported：被导入**非子元素**（packages、modules、attrs）
	-	可以通过`from`语句导入
	-	`import`导入`ModuleNotFoundError`暗示，`import`是
		是按照**文件**处理（也解释`import`无法导入attrs）

####	`import`

```python
import sys
import os, sys, time
	# 方便，但违背python风格指南
	# 建议每个导入语句单独成行
import sys as system
	# 重命名模块
	# 重复`import sys`模块不会多次执行，使用`reload`
```

####	`from`

```python
from functools import lru_cache
	# 导入模块
from os import *
	# 方便，但会打乱命名空间
from os import (path, walk, unlink,
			uname, remove, rename)
	# 用圆括号括起可以分行
from os import path, walk, unlink\
			, uname, remove, rename
	# 否则使用`\`续行符
```

####	`.`相对导入

使用`.`表示相对导入包、模块，避免偶然情况下同标准库中的模块
产生冲突

-	适用于相关性强、放入包中的代码
-	脚本模式（在命令行中执行`.py`文件）不支持相对导入，若
	想要执行包中某个模块（含有相对导入），需将包添加进python
	检索路径
-	要跨越多个文件层级导入，只需要使用多个`.`，但是PEP 328
	建议，相对导入层级不要超过两层

```python

'''
__init__.py
'''
from . import subpackage1
from . import subpackage2

'''
subpackage1/__init__.py
'''
from . import module_x
from . import module_y

'''
subpackage1/module_x.py
'''
from . module_y import spam as ham
def main():
	ham()

if __name__ == "__main__":
	main()
	# `$ python module_x.py`报错，脚本模式不支持相对导入

'''
subpackage1/module_y.py
'''
def spam():
	print("spam" * 3)
```

####	可选导入（Optional Imports）

希望优先使用某个模块、包，同时也希望在其不存在的情况下有备选

```python
try:
	# for python3
	from http.client import responses
except ImportError:
	try:
		# for python2.5-2.7
		from httplib import responses
	except:
		# for python2.4
		from BaseHTTPServer import BaseHTTPResponseHandler as _BHRH
		responses = dict([(k, v[0]) for k, v in _BHRH response.items()])

try:
	from urlparse import urljoin
	from urllib2 import urlopen
except ImportError:
	# for python3
	from urllib.parse imoprt urljoin
	from urllib.request import urlopen
```

####	局部导入

在局部作用域中导入模块，不经常使用的函数使用局部导入比较合理
，但是根据约定，所有的导入语句都应该为模块顶部

```python
import sys

def square_root(a):
	# import to local scope
	import math
	return math.sqrt(a)
```

###	导入检索路径（Import Search Path）

导入模块（包）时，解释器首先搜索**内置**模块，没有找到则在
`sys.path`给出的目录列表中搜索模块，`sys.path`以以下位置
初始化（即内置模块优先级高于任何其他）

-	输入脚本目录（或当前目录）
-	`PYTHONPATH`环境变量
-	安装相关默认值

因此，修改`sys.path`即可更改python导入逻辑

####	动态增加路径

```python
import sys
sys.path.insert(1, /path/to/fold/contains/module)
	# 临时生效，对不经常使用的模块较好
```

####	修改`PYTHONPATH`环境变量

```shell
 # .bashrc
export PYTHONPATH=$PYTHONPATH:/path/to/fold/contains/module
```

-	对许多程序都使用的模块可以采取此方式
-	会改变所有Python应用的搜索路径

####	增加`.pth`文件

在`/path/to/python/site-packages`（或其他查找路径目录）下
添加`.pth`配置文件，内容为需要添加的路径

```conf
 # extras.pth
/path/to/fold/contains/module
```

-	简单、推荐
-	python在遍历已知库文件目录过程中，遇到`.pth`文件会将其中
	路径加入`sys.path`中

###	具体模块

####	`sys.setdefaultencoding`

直接查看`sys`模块中并没有这个方法

-	py2中因为这个方式是为site模块准备的，调用之后从sys模块
	空间抹除，需要reload(sys)才能调用

-	py3中好像直接移除这个方法了

##	Python继承实现

###	MRO列表

-	对定义的每个类，python会计算一个所谓的方法解析顺序
	（MRO）列表

	-	一个简单的包含所有基类的线性顺序表
	-	可以通过**类**的`__mro__`属性访问

-	MRO列表是通过C3线性化算法实现的，其合并所有父类的MRO
	列表，并遵循如下3条原则

	-	子类先于父类检查
	-	多个父类会根据其在列表中的顺序被检查
	-	如果对下一个类存在多个合法的选择，选择第一个父类

-	为了实现继承，python会在MRO列表上从左到右开始查找基类，
	直到第一个匹配这个属性的类为止

###	`super`

-	调用`super`函数时，python会在MRO表中上继续搜索下个类
	，只要每个重定义的方法统一使用`super`并调用一次，
	控制流会遍历整个MRO列表，每个方法也只会调用一次

	-	所以，在类继承中，尽量使用`super`函数调用超类方法，
		而不要直接使用超类调用方法，否则可能会多次调用同一个
		超类的方法
	-	事实上，`super`并不一定查找到的是某个类在MRO中的下个
		直接父类，甚至可以不是父类

-	因为`super`方法可能调用的不是想要的方法，所以需要遵循
	以下原则

	-	继承体系中，所有相同名字的方法拥有可兼容的参数名，
		比如：相同参数个数、名称
	-	最好确保最顶层类提供这个方法的实现，这样保证MRO上的
		查找链肯定可以找到某个方法

```python
class A:
	def spam(self):
		print("A.spam")
		super().spam()
			# 类`A`没有含有`spam`方法的父类

class B:
	def spam(self):
		print("B.spam")

class C(A, B):
	pass

def test():
	c = C()
	c.spam()
		# `C`从`A`继承`spam`方法
		# `A`中的`spam`方法中的`super`调用了其非父类的类`B`
			# 的`spam`方法
```

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
