#	Python对象、值、类型

##	对象、值、类型

对象：python中对数据的抽象

-	python中所有数据都是由对象、对象间关系表示
	-	按冯诺依曼“存储程序计算机”，代码本身也是由对象表示

-	每个对象都有各自**编号**、**类型**、**值**
	-	编号：可以视为对象在内存中地址，对象创建后不变
		-	`id()`：获取代表对象编号的整形
		-	`is`：比较对象编号是否相同
	-	类型：决定对象支持的操作、可能取值，对象创建后不变
		-	`type`：返回对象类型
	-	值：对象值可变性由其类型决定
		-	可变的：值可以改变的对象
		-	不可变的：值（直接包含对象编号）不可改变的对象

	> - CPython：`id(x)`返回存放`x`的地址

-	对象不会被显式销毁
	-	无法访问时**可能**被作为垃圾回收
		-	运行具体实现推迟垃圾回收或完全省略此机制
		-	实现垃圾回收是质量问题，只要可访问对象不会被回收
			即可
	-	以下情况下，正常应该被回收的对象可能继续存活
		-	使用实现的跟踪、调试功能
		-	通过`try...except...`语句捕捉异常
	-	不要依赖不可访问对象的立即终结机制
		-	应当总是显式关闭外部资源引用

	> - CPython：使用带有（可选）**延迟检测循环链接垃圾**的
		引用计数方案
	> > -	对象**不可访问**时立即回收其中大部分，但不保证
			回收包含**循环引用**的垃圾

-	类型会影响对象行为几乎所有方面
	-	得到新值的运算
		-	不可变类型：可能返回同类型、目标取值现有对象引用
		-	可变类型：不允许返回已存在对象
		-	`a = b = 1`后，`a`、`b`可能指向相同对象
			（取决于具体实现）

	> - CPython：相同整形值都引用同一个对象

##	标准类型层级结构

###	`None`

-	`NoneType`只有一种取值，`None`是具有此值的唯一对象
	-	通过内置名称`None`访问
-	多数情况表示空值
	-	未显式指明返回值函数返回`None`
-	逻辑值：假

###	`NotImplemented`

-	`NotImplementedType`只有一种取值，`NotImplemented`是具有
	此值的唯一对象
	-	通过内置名称`NotImplemented`访问
-	数值、扩展比较方法在操作数没有该实现操作时应该返回此值
	-	解释器会根据指定运算符尝试反向运算、其他回退操作
-	逻辑值：真

###	`Ellipsis`

-	`ellipsis`只有一种取值，`Ellipsis`是具有此值的唯一对象
	-	通过字面值`...`、内置名称`Ellipsis`访问
-	逻辑值：真

###	`numbers.Number`

-	由数字字面值创建，被作为算法运算符、算数内置函数返回结果
-	不可变：一旦创建其值不再改变

####	`numbers.Integral`

`numbers.Integral`：表示数学中整数集合

-	`int`：整形，表示任意大小数字，仅受限于可用内存
	-	变换、掩码运算中以二进制表示
	-	负数以2的补码表示，类似符号位向左延伸不满空位

-	`bool`：布尔型，表示逻辑值真、假
	-	`True`、`False`是唯二两个布尔对象
	-	整形子类型：在各类场合中行为类似整形`1`、`0`，仅在
		转换为字符串时返回`"True"`、`"False"`

####	`numbers.Real`

`float`：表示机器级双精度浮点数

-	接受的取值返回、溢出处理取决于底层结构、python实现
-	python不支持单精度浮点
	-	没必要因为节省处理器、内存消耗而增加语言复杂度

####	`numbers.Complex`

`complex`：以一对机器级双精度浮点数表示复数值

-	实部、虚部：可通过只读属性`z.real`、`z.imag`获取

###	序列

序列：表示以非负整数作为索引的**有限有序集**

-	`len`：返回序列条目数量
-	`a[i:j]`：支持切片
	-	序列切片：与序列类型相同的新序列，索引从0开始
	-	某些序列支持`a[i:j:step]`扩展切片

####	不可变序列

-	不可变序列类型对象一旦创建不能改变
-	若包含其他可变对象引用，则可变对象“可改变”

#####	`str`

字符串：由Unicode码位值组成序列
#todo？？？不是utf16编码方案
-	范围在`U+0000 ~ U+10FFFF`内所有码位值均可在字符串中
	使用
-	字符串中每个字符为一个长度为1的字符串对象
-	相关函数、方法
	-	`ord`：转换单个字符字符串为（整形）码位
	-	`chr`：转换（整形）码位为单个字符字符串
	-	`str.encode`：使用指定编码方案编码为`bytes`

#####	`tuple`

元组：

-	元组中条目可以是任意python对象
-	创建元组
	-	包含两个、或以上条目的元组：由逗号分隔表达式创建
	-	单项元组：须在表达式后加逗号创建
	-	空元组：通过内容为空的圆括号创建

#####	`bytes`

字节串：不可变数组

-	每个条目都是8位字节
	-	取值范围`0~255`
-	创建字节串
	-	`b'abc'`：字节串字面值
	-	`bytes()`：构造器
-	相关函数、方法
	-	`.decode`：解码为相关字符串

####	可变序列

-	可变序列在被创建后仍可被改变
-	下标、切片标注可以作为赋值、`del`语句的目标
-	不可哈希

> - `array`、`collections`模块提供额外可变序列类型

#####	`list`

列表

-	列表中条目可以是任意python对象
-	列表由方括号括起、逗号分隔的多个表达式构成
	-	创建长度为0、1的列表没有特殊规则

#####	`bytearray`

字节数组

-	除是可变之外，其他方面同不可变`bytes`对象一致
-	创建字节数组
	-	`bytearray()`构造器

###	集合类型

-	表示**不重复**、**不可变**对象组成的无序、有限集合
	-	不能通过下标索引
	-	可以迭代
	-	可以通过内置函数`len`返回集合中条目数量

-	python中集合类似`dict`通过hash实现
	-	集合元素须遵循同字典键的不可变规则
	-	数字：相等的数字`1==1.0`，同一集合中只能包含一个

-	常用于
	-	快速成员检测、去除序列中重复项
	-	进行交、并、差、对称差等数学运算

####	`set`

集合：可变集合

-	创建集合
	-	`set()`内置构造器
-	相关函数、方法
	-	`.add`：添加元素

####	`frozenset`

冻结集合：不可变集合

-	创建集合
	-	`frozenset()`内置构造器
-	`frozenset`对象不可变、可哈希，可以用作另一集合的元素、
	字典的键

###	映射

映射：表示任何索引集合所索引的对象的集合

-	通过下标`a[k]`可在映射`a`中选择索引为`k`的条目
	-	可在表达式中使用
	-	可以作为赋值语句、`del`语句的目标
-	`len()`可返回映射中条目数

> - `dbm.ndbm`、`dbm.gnu`、`collections`模块提供额外映射

####	`dict`

字典：表示可由**几乎任意值作为索引**的有限个对象集合

-	字典的高效实现需要使用键hash值以保持一致性
	-	不可作为键的值类型
		-	包含列表、字典的值
		-	其他通过对象编号而不是值比较的可变对象
	-	数字：相等的数字`1==1.0`索引相同字典条目

-	字典可变

-	创建字典
	-	`{...}`标注创建

###	可调用类型

-	可以被用于函数调用操作

####	用户定义函数

用户定义函数：通过函数定义创建

-	函数对象支持获取、设置任意属性
	-	用于给函数附加元数据
	-	使用属性点号`.`获取、设置此类属性

#####	元属性

-	`__doc__`：函数的文档字符串
	-	没有则为`None`
	-	不会被子类继承
-	`__name__`：函数名称
-	`__qualname__`：函数的*qualified name*
-	`__module__`：函数所属模块名称
	-	没有则为`None`
-	`__defaults__`：有默认值参数的默认值组成元组
	-	没有具有默认值参数则为`None`
-	`__code__`：**编译后**函数体代码对象
-	`__globals__`：存放函数中全局变量的字典的引用
	-	即引用函数所属模块的全局命名空间
	-	只读
-	`__dict__`：命名空间支持的函数属性
-	`__closure__`：包含函数自由变量绑定cell的元组
	-	没有则为`None`
	-	只读
-	`__annotations__`：包含参数注释的字典
	-	字典键为参数名、`return`（若包含返回值）
-	`__kwdefaults__`：`keyword-only`参数的默认值字典

> - 大部分可写属性会检查赋值类型

####	实例方法

实例方法：用于结合类、类实例、任何可调用对象

-	方法可以获取（只读）底层函数对象的任意属性
-	方法对象创建时间：访问类的用户定义函数对象、类方法对象时
	-	通过实例获取用户定义函数对象时创建实例方法对象
		-	`__self__`属性：为类实例，方法对象称为被绑定
		-	`__func__`属性：原始函数对象
	-	通过类、实例获取其他方法对象时创建用户定义方法对象
		-	行为同函数对象一致
		-	新方法实例`__func__`属性不是原始方法对象，而是
			其`__func__`属性
	-	通过类、实例获取类方法对象时创建实例方法对象
		-	`__self__`属性：类
		-	`__func__`属性：底层函数对象
-	调用实例方法对象时
	-	方法对象底层`__func__`函数被调用
	-	`__self__`类实例插入函数参数列表开头
-	派生于类方法对象的实例方法对象中
	-	`__self__`是类本身
-	函数对象到实例方法对象的转换每次获取实例该属性时都会发生
	-	某些情况下，将属性赋值给本地变量、调用是一种高效的
		优化方法
	-	非用户定义函数、不可调用对象在被获取时不会发生转换
	-	类实例属性的用户定义函数不会被转换为绑定方法，仅在
		函数是类的属性时才会发生

#####	特殊元属性

-	`__self__`：类对象实例
-	`__func__`：函数对象实例
-	`__doc__`：方法文档，等同于`__func__.__doc__`
-	`__name__`：方法名，等同于`__func__.name__`
-	`__module__`：定义方法所在模块名

####	*Generator Functions*

生成器函数：使用`yield`语句的函数、方法称为生成器函数

-	生成器函数调用时返回迭代器对象，用于执行函数体
	-	`.__next__()`方法将执行函数直到`yield`语句提供值
	-	执行`return`语句、函数体执行完毕将
		`raise StopIteration`

####	*Coroutine Functions*

协程函数：使用`async def`定义的函数、方法

-	调用时返回一个`coroutine`对象
	-	可能包含`await`表达式、`async with`、`async for`语句

####	*Asynchronous Generator Functions*

异步生成函数：使用`async def`定义、包含`yield`语句的函数、
方法

-	调用时返回异步生成器对象，在`async for`语句用于执行
	函数体
	-	`.__anext__()`方法将返回`awaitable`，该对象在等待时
		将执行直到使用`yield`表达式输出值
	-	执行到空`return`语句、函数体执行完毕后将
		`raise StopAsyncIteration`

####	*Built-in Functions*

内置函数：对C函数的外部封装

-	参数数量、类型由C函数决定

#####	特殊元属性

-	`__doc__`：函数文档字符串，只读
-	`__name__`：函数名称
-	`__self__`：`None`
-	`__module__`：函数所属模块名称


####	*Built-in Method*

内置方法：实际上是内置函数另一种形式，包含传入C函数的对象
作为隐式额外参数

#####	特殊元属性

-	`__self__`：只读，对象实例

####	*Classes*

类：类对象通常作为“工厂”创建自身实例

####	*Class Instances*

类实例：在所属类中定义`__call__()`方法即成为可调用对象

###	*Module*

模块：python代码的基本组织单元

-	导入系统创建
	-	`import`语句
	-	`importlibd.import_module()`、`__import__()`函数
-	模块对象具有由字典`__dict__`实现的命名空间
	-	属性引用：被转换为该字典中查找`m.__dict__['x']`
	-	属性赋值：更新模块命名字典空间
	-	不包含用于初始化模块的代码对象
	-	模块中定义函数`__globals__`属性引用其

####	元属性

-	`__name__`：模块名称
-	`__doc__`：模块文档字符串
-	`__annotaion__`：包含变量标注的字典
	-	在模块体执行时获取
-	`__file__`：模块对应的被加载文件的路径名
	-	若加载自一个文件，某些类型模块可能没有
		-	C模块静态链接至解释器内部
	-	从共享库动态加载的扩展模块，该属性为共享库文件路径名
-	`__dict__`：以字典对象表示的模块命名空间

> - CPython：由于CPython清理模块字典的设定，模块离开作用域时
	模块字典将被清理，即使字典还有活动引用，可以复制该字典、
	保持模块状态以直接使用其字典

###	*Custom Classes*

自定义类：通过类定义创建

-	每个类通过字典对象`__dict__`实现独立的命名空间
	-	类属性引用被转化为在此字典中查找
	-	其中未发现属性名时，继续在基类中查找
		-	基类查找使用C3方法解析顺序，即MRO列表
	-	也存在一些**钩子对象**允许其他定位属性的方式

-	当类属性引用*yield*类方法对象时，其将转化为`__self__`
	属性为当前类对象的**实例方法对象**

-	当类属性引用*yield*静态方法对象时，其将转换为静态方法
	对象所封装的对象

-	类属性复制会更新类字典，不会更新基类字典

-	类对象可被调用产生类实例

-	`__bases__`：包含基类的元组，依在基类列表中出现的顺序

###	*Class Instances*

类实例：通过**调用类对象**创建

-	每个类实例都有一个通过字典对象`__dict__`实现的独立命名
	空间
	-	属性引用首先在此字典中查找
	-	其中未发现属性名时，继续在对应类属性中查找
		-	用户定义函数对象：其会被转化为**实例方法对象**
			-	`__self__`属性即为该实例
		-	静态方法、类方法对象：同样会被转化

		> - 描述器属性有特殊处理，实际存放在类`__dict__`中
			对象不同
	-	若未找到类属性，对象对应类具有`__getattr__()`方法，
		将调用该方法

-	属性赋值、删除会更新实例字典，不会更新对应类字典
	-	若类具有`__setattr__`、`__delattr__`方法，将调用方法
		而不是直接更更新对应实例字典

####	特殊元属性

-	`__class__`：实例对应类

###	I/O对象/文件对象

文件对象：表示打开的文件

-	创建文件对象
	-	`open()`内置函数
	-	`os.popen()`、`os.fdopen()`
	-	`socket.makefile()`

-	`sys.stdin`、`sys.stdout`、`sys.stderr`会初始化为对应于
	解释器的标准输入、输出、错误流对象
	-	均以文本模式打开
	-	遵循`io.TextIOBase`抽象类所定义接口

###	内部类型

-	解释器内部使用的类型
-	定义可能解释器版本更新而变化

####	*Code Object*

代码对象：编译为字节的可执行python代码，也称*bytecode*

-	代码对象和函数对象区别
	-	代码对象不包含上下文；函数对象包含对函数全局对象
		（函数所属模块）的显式引用
	-	函数对象中存放默认参数值
	-	代码对象不可变，也不包含对可变对象的应用

#####	特殊属性

-	`co_name`：函数名称
-	`co_argcount`：位置参数数量
-	`co_nlocals`：函数使用的本地变量数量（包括参数）
-	`co_varnames`：包含本地变量名称的元组
-	`co_freevars`：包含自由变量的元组
-	`co_code`：表示字节码指令序列的字符串
-	`co_consts`：包含字节码所使用的字面值元组
	-	若代码对象表示一个函数，第一项为函数文档字符，没有
		则为`None`
-	`co_names`：包含字节码所使用的名称的元组
-	`co_filenames`：被编译代码所在文件名
-	`co_firstlineno`：函数首行行号
-	`co_lnotab`：以编码表示的字节码偏移量到行号映射的字符串
-	`co_stacksize`：要求栈大小（包括本地变量）
-	`co_flags`：以编码表示的多个解释器所用标志的整形数
	-	`0x04`位：函数使用`*arguments`接受任意数量位置参数
	-	`0x08`位：函数使用`**keywords`接受任意数量关键字参数
	-	`0x20`位：函数是生成器
	-	`0x2000`位：函数编译时使用启用未来除法特性
	-	其他位被保留为内部使用

###	*Frame Objects*

栈帧对象：执行帧

-	可能出现在回溯对象中，还会被传递给注册跟踪函数

####	特殊只读属性

-	`f_back`：前一帧对象，指向主调函数
	-	最底层堆栈帧则为`None`
-	`f_code`：此栈帧中所执行的代码对象
-	`f_locals`：查找本地变量的字典
-	`f_globals`：查找全局变量
-	`f_builtins`：查找内置名称
-	`f_lasti`：精确指令，代码对象字节码字符串的索引

####	特殊可写属性

-	`f_trace`：`None`，或代码执行期间调用各类事件的函数
	-	通常每行新源码触发一个事件
-	`f_trace_lines`：设置是否每行新源码触发一个事件
-	`f_trace_opcodes`：设置是否允许按操作码请求事件
-	`f_lineno`：帧当前行号
	-	可以通过写入`f_lineno`实现Jump命令

####	方法

-	`.clear()`：清楚该帧持有的全部对本地变量的引用
	-	若该栈帧为属于生成器，生成器被完成
	-	有助于打破包含帧对象的循环引用
	-	若帧当前正在执行则会`raise RuntimeError`

###	*Traceback Objects*

回溯对象：表示异常的栈跟踪记录

-	异常发生时会隐式创建回溯对象
	-	查找异常句柄使得执行栈展开时，会在每个展开层级的当前
		回溯之前插入回溯对象
	-	进入异常句柄时，栈跟踪将对程序启用
	-	获取：`sys.exc_info()`返回的元组第三项、异常的
		`__traceback__`属性
	-	程序没有合适的处理句柄时，栈跟踪将写入标准错误

-	可通过`types.TracebackType`显式创建
	-	由回溯对象创建者决定如何链接`tb_next`属性构成完整
		栈追踪

####	特殊只读属性

-	`tb_frame`：执行当前层级的执行栈帧
-	`tb_lineno`：给出发生异常所在行号
-	`tb_lasti`：最后具体指令

> - 若异常出现在没有匹配的`except`子句、没有`finally`子句
	的`try`中，回溯对象中的行号、最后指令可能于相应帧对象中
	行号不同

###	*Slices Object*

切片对象：表示`__getitem__()`方法得到的切片

-	可以使用内置的`slice()`函数创建

####	特殊只读属性

-	`start`：下界
-	`stop`：上界
-	`step`：步长值

> - 属性可以具有任意类型

####	方法

-	`.indices(self, length)`：计算切片对象被应用到`length`
	长度序列时切片相关信息
	-	返回值：`(start, stop, step)`三元组
	-	索引号缺失、越界按照正规连续切片方式处理

###	*Static Method Objects*

静态方法对象：对任意其他对象的封装，通常用于封装用户定义方法
对象

-	提供避免**将函数对象转换为方法对象**的方式
-	从类、类实例获取静态方法对象时，实际返回的是封装的对象，
	不会被进一步转换
-	静态方法对象自身不是可调用的，但器封装的对象同上可调用
-	通过内置`staticmethod()`构造器创建

###	*Class Method Objects*

类方法对象：

-	和静态方法一样是对其他对象的封装，会改变从类、类实例
	获取该对象的方式
-	通过`classmethod`构造器创建

####	特殊可写属性

-	`tb_next`：栈跟踪中下一层级（通往发生异常的帧），没有
	下一层级则为`None`

##	类属性控制

###	封装属性

python没有访问控制，不依赖语言特性封装数据，而是遵循一定属性
、方法命名规约达到效果

-	任何以单`_`开头的名字都应该是内部实现
	-	此约定同样适用于模块名、模块级别函数
	-	python不会真的防止访问内部名称，但是这样会导致脆弱的
		代码

-	以`__`开头的名称会使名称变为其他形式
	-	目的是为了防止继承，这样的属性无法通过继承覆盖
	-	如果清楚代码会涉及子类，且应该在子类中隐藏起来，考虑
		使用双下划线开头

-	以单`_`结尾，避免定义的变量和保留关键字冲突

```python
class A:
	def __init__(self):
		self._internal = 0
			# private attr
		self.public = 1

	def public_method(self):
		pass

	def _internal_method(self):
		# private method
		pass

class B:
	def __init__(self):
		self.__private = 0
			# 将会被重命名为`_B__private`

	def __private_method(self):
		# 将会被重命名为`_B__private_method`
		pass

	def public_method(self):
		self.__private_method()

def C(B):
	def __init__(self):
		super().__init__()
		self.__private = 1
			# 不会覆盖`B.__private`
			# 会被重命名为`C._C__private`

	def __private_method(self):
		# 不会覆盖`B.__private_method`
		# 会被重命名为`C._C__private_method`
		pass
```

###	`@property`可管理属性

使用`@property`装饰器可以为类的属性增加其他处理逻辑，如：
类型检查、合法性验证

-	`@property`装饰器类将getter-like方法转变为属性，然后设置
	3个方法名开头装饰器分别装饰其他两个**同名方法**

	-	`@method.setter`：将setter-like方法转换成`method`
		属性，在对`.method`赋值时将被调用
		-	不设置时会将方法转换为“只读属性”
	-	`@method.deleter`：将deleter-like方法转换为`method`
		属性，在`del(.method)`时将被调用
	-	`@method.getter`：将getter-like方法转换为`method`
		属性，在访问`.method`时调用，这个一般只在子类仅修改
		父类property属性的getter方法时使用

-	`@property`装饰的属性和普通属性没有区别

	-	但是在访问时会自动触发`getter`、`setter`、`deleter`
	-	property属性就是绑定有这些方法的类实例，通常不需要
		手动调用这些方法
		-	`instance.method.fget`
		-	`instance.method.fset`
		-	`instance.method.fdel`
	-	没有必要却使用`@property`时会迷惑读者、使代码臃肿、
		降低代码效率
	-	有时候使用get/set方法可能有更好的兼容性

-	应用`@property`时，仍然需要在类中存储底层数据，在
	`getter`、`setter`会对其进行处理

	-	初始化时，不应该直接设置底层数据属性，会绕过`setter`
		的参数检查
	-	当然，有的写property属性没有对应的底层数据属性，是在
		需要的时候计算出来的，当然也没有对应的`setter`方法
		实现

-	可以直接使用已有getter-like、setter-like方法创建
	property类实例，定义property属性

	-	property属性就是一个类，为其实现`setter`、`getter`、
		`deleter`方法，然后作为**类属性**即可
	-	因为本身直接使用`@property`装饰方法也是相当于直接
		设置的是**类属性**
	-	其内部有可能是使用的描述器将值记录在实例底层字典
		`__dict__`中

```python
class Student(object):

	def __init__(self, value):
		self.birth = value
			# 使用`self.birth`而不是`self._birth`，保证即使
				# 实在初始化时仍然进行参数检查

	@property
		# 将一个getter-like方法变为属性
		# `@property`同时会创建装饰器`@method.setter`
	def birth(self):
		return self._birth

	@birth.setter
		# `@property`对应，将setter-like方法变为属性
	def birth(self, value):
		if not instance(value, int):
			raise ValueError("birth must be an integer")
		if value < 1900 or value > 2020:
			raise ValueError("birth must between 1900 ~ 2020")
		self._birth = value

	@birth.deleter
		# 同`@property`对应，在`del`时调用
	def birth(self):
		del(self._age)
		del(self._birth)

	@property
		# 只设置`@property`而没有设置对应`@birth.setter`
		# 这样`birth`就成了只读属性
	def age(self):
		return 2018 - self._birth

	def get_first_name(self):
		return self._first_name
	
	def set_first_name(self):
		if not instance(value, str):
			raise TypeError("expected a string")
		self._first_name = value

	def del_first_name(self):
		raise AttributeError("can't delete attribute")

	name = property(get_first_name,
		set_first_name,
		del_first_name)
		# 在已有getter-like、setter-like方法上创建property
		# 注意：这里就是应该定义类属性，本身使用`@property`
			# 装饰器也是相当于创建类属性
```

###	`classmethod`类方法

###	`staticmethod`静态方法

###	构造实例

-	`@classmethod`类方法可以实现多个构造器

-	`__new__`可以方法绕过`__init__`，创建未初始化的实例

	-	这种方法可以用于**反序列**对象，从字符串反序列构造
		符合要求的对象

```python
import time

class Date:
	def __init__(self, year, month, day):
		self.year = year
		self.month = month
		self.day = day

	@classmethod
	def today(cls):
		# 类方法接受`class`作为第一个参数
		# 和普通属性方法一样能够继承
		t = time.localtime()
		return cls(t.tm_year, t.tm_mon, t.tm_today)

	@classmethod
	def today2(cls):
		d = cls.__new__(cls)
			# 使用`__new__`绕过`__init__`创建新实例，不对
				# 实例进行初始化
		t = localtime()
		d.year = t.tm_year
		d.month = t.tm_mon
		d.day = t.tm_mday
		return d

class NewData(Date):
	pass

def test():
	c = Date.today()
	d = NewDate.today()
```

