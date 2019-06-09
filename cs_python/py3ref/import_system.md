#	Python包、模块

##	综述

###	导入方式

*importing*操作可以使得模块能够访问其他模块中代码

-	`import`：结合了以下两个操作，发起导入调机制最常用方式

	-	搜索指定名称模块：对`__import__()`带有适当参数调用
	-	将搜索结果绑定到当前作用域中名称：`__import__`返回值
		被用于执行名称绑定操作

-	`__import__()`：只执行模块搜索、找到模块后创建`module`
	-	可能产生某些副作用
		-	导入父包
		-	更新各种缓存：`sys.modules`

-	`importlib.import_module()`

	-	可能会选择绕过`__import__`，使用其自己的解决方案实现
		导入机制
	-	用于为动态模块导入提供支持

	> - `importlib`模块参见标准库

####	子模块

-	任意机制加载子模块时，父模块命名空间中会添加对子模块对象
	的绑定

###	*Packages*

-	python只有一种模块对象类型：所有模块都属于该类型，C、
	Python语言实现均是

-	包：为帮助组织模块、并提供名称层次结构引入
	-	可将包**视为**文件系统中目录、模块视为目录中文件，
		但**包、模块不是必须来自文件系统**
	-	类似文件系统，包通过层次结构进行组织：包内包括模块、
		子包

-	**所有包都是模块，但并非所有模块都是包**
	-	包是一种特殊的模块
	-	特别的，任何具有`__path__`属性的模块都被当作包

-	所有模块都有自己的名字
	-	子包名与其父包名以`.`分隔，同python标准属性访问语法

####	*Regular Packages*

正规包：通常以包含`__init__.py`文件的目录形式出现

-	`__init__.py`文件可以包含和其他模块中包含python模块相似
	的代码

-	正规包被导入时
	-	`__init__.py`文件会隐式被执行，其中定义对象被绑定到
		该包命名空间中名称
	-	python会为模块添加额外属性

####	*Namespace Packages*

命名空间包：由多个部分构成，每个部分为父包增加一个子包

-	包各部分可以物理不相邻，不一定直接对应到文件系统对象，
	可能是无实体表示的虚拟模块
	-	可能处于文件系统不同位置
	-	可能处于zip文件、网络上，或在导入期间其他可搜索位置

-	`__path__`属性不是普通列表，而是定制的可迭代类型
	-	若父包、或最高层级包`sys.path`路径发生改变，对象会
		在包内的下次导入尝试时，自动执行新的对包部分的搜索

-	命名空间包中没有`__init__.py`文件
	-	毕竟可能有多个父目录提供包不同部分，彼此物理不相邻
	-	python会在包、子包导入时为其创建命名空间包

###	导入相关模块属性

> - 以下属性在加载时被设置，参见
	*cs_python/py3ref/import_system*

-	`__name__`：模块完整限定名称，唯一标识模块

-	`__loader__`：导入系统加载模块时使用的加载器对象
	-	主要用于内省
	-	也可用于额外加载器专用功能

-	`__package__`：取代`__name__`用于主模块计算显式相对
	导入
	-	模块为包：应设置为`__name__`
	-	非包：最高层级模块应设为空字符串，否则为父包名

	> - 预期同`__spec__.parent`值相同，未定义时，以
		`__spec__.parent`作为回退项

####	`__spec__`

`__spec__`：导入模块时要使用的**模块规格说明**

-	对`__spec__`正确设置将同时作用于解释器启动期间
	初始化的模块
-	仅`__main__`某些情况下被设置为`None`

####	`__path__`

-	**具有该属性模块即为包**：包模块必须设置`__path__`属性，
	非包模块不应设置

-	在导入子包期间被使用，在导入机制内部功能同`sys.path`，
	即用于提供模块搜索位置列表

	-	但受到更多限制，其必须为字符串组成可迭代对象，但若其
		没有进一步用处可以设置为空
	-	适用作用于`sys.path`的规则
	-	`sys.path_hooks`会在遍历包的`__path__`时被查询

-	可在包的`__init__.py`中设置、更改
	-	在*PEP420*引入之后，命名空间包不再需要提供仅包含操作
		`__path__`代码的`__init__.py`文件，导入机制会自动为
		命名空间包正确设置`__path__`
	-	在之前其为实现命名空间包的典型方式

####	`__repr__`

-	若模块具有`__spec__`，导入机制将尝试使用其中规格信息生成
	repr

	-	`name`
	-	`loader`
	-	`origin`
	-	`has_location`

-	若模块具有`__file__`属性，将被用作repr的一部分

-	否则若模块具有`__loader__`属性且非`None`，则加载器repr
	将被用作模块repr的一部分

-	其他情况下，仅在repr中适用模块的`__name__`

> - 可以在模块规则说明中显式控制模块对象repr

####	`__file__`/`__cached__`

> - `__file__`：模块对应的被加载文件的路径名
> - `__cached__`：编译版本代码（字节码文件）路径

-	`__file__`为可选项，须为字符串
	-	可以在其无语法意义时不设置
	-	对从共享库动态加载的扩展模块，应为共享库文件路径名

-	`__cached__`
	-	不要求编译文件已经存在，可以表示**应该存放**编译文件
		的位置
	-	不要求`__file__`已经设置
		-	有时加载器可以从缓存加载模块但是无法从文件加载
		-	加载静态链接至解释器内部的C模块

-	从`.pyc`文件加载缓存字节码前会检查其是否最新
	-	默认通过比较缓存文件中保存的源文件修改时间戳实现
	-	也支持基于哈希的缓冲文件，此时`.pyc`文件中保存源文件
		哈希值
		-	检查型：求源文件哈希值再和缓存文件中哈希值比较
		-	非检查型：只要缓存文件存在就直接认为缓存文件有效

	> - `--check-hash-based-pycs`命名行选项设置基于哈希的
		`.pyc`文件有效性

###	执行相关模块属性

-	`__doc__`：模块文档字符串
-	`__annotaion__`：包含变量标注的字典
	-	在模块体执行时获取
-	`__dict__`：以字典对象表示的模块命名空间

> - CPython：由于CPython清理模块字典的设定，模块离开作用域时
	模块字典将被清理，即使字典还有活动引用，可以复制该字典、
	保持模块状态以直接使用其字典

###	`sys.modules`模块缓存

`sys.modules`映射：缓存之前导入的所有模块（包括中间路径）
（即导入子模块会注册父模块条目）

-	其中每个键值对就是限定名称、模块对象

-	在其中查找模块名称
	-	若存在需要导入模块，则导入完成
	-	若名称对应值为`None`则`raise ModuleNotFoundError`
	-	若找不到指定模块名称，python将继续搜索

-	映射可写，可删除其中键值对
	-	不一定破坏关联模块，因为其他模块可能保留对其引用
	-	但是会使**命名模块**缓存条目无效，导致下次导入时重新
		搜索命名模块，得到两个不同的两个模块对象

	> - `importlib.reload`将重用相同模块对象，通过重新运行
		模块代码重新初始化模块内容

##	*Finders And Loaders*

> - *Finders*：查找器，确定能否使用所知策略找到指定名称模块
> - *Loaders*：加载器，加载找到的指定模块
> - *Importer*：导入器，同时实现两种接口的对象，在确定能加载
	所需模块时会返回自身

-	导入机制通过*import hooks*实现扩展
	-	可以加入新的查找器以扩展模块搜索范围、作用域

-	工作流程：在`sys.modules`缓存中无法找到指定名称模块时
	-	查找器若能找到指定名称模块，返回模块规格说明*spec*
	-	加载器将利用查找器返回的模块规格说明加载模块

###	*Import Path*

导入路径：文件系统路径、zip文件等*path term*组成的位置列表

-	其中元素不局限于文件系统位置，可扩展为字符串指定的任意
	可定位资源
	-	URL指定资源
	-	数据库查询

-	位置条目来源
	-	通常为`sys.path`
	-	对次级包可能来自上级包的`__path__`属性

-	其中每个路径条目指定一个用于搜索模块的位置
	-	*path based finder*将在其中查找导入目标

####	`sys.path`

`sys.path`：模块、包搜索位置的字符串列表

-	初始化自`PYTHONPATH`环境变量、特定安装和实现的默认设置、
	执行脚本目录（或当前目录）

-	其中条目可以指定文件系统中目录、zip文件、可用于搜索模块
	的潜在位置

-	只能出现字符串、字节串，其他数据类型被忽略
	-	字节串条目使用的编码由导入路径钩子、
		*path entry finder*确定

> - 所以可以修改`sys.path`值定制导入路径，CPython实现参见
	*cs_python/py3ref/import_system*

####	`sys.path_import_cache`

`sys.path_importer_cache`：存放路径条目到路径条目查找器映射
的缓存

-	减少查找路径条目对应路径条目查找器的消耗，对特定路径条目
	查找对应路径条目查找只需进行一次

-	可从中移除缓存条目，以强制基于路径查找器执行路径条目搜索

###	*Import Hooks*

-	*meta hooks*：元[路径]钩子
	-	导入过程开始时被调用，此时仅`sys.modules`缓存查找
		发生，其他导入过程未发生
	-	所以允许元钩子重载`sys.path`过程、冻结模块甚至内置
		模块

	> - **元钩子即导入器/元路径查找器**
	> - `sys.meta_path`为元路径查找器列表，可在其中注册定制
		元钩子

-	*path[ entry] hooks*：导入路径钩子
	-	是`sys.path`、`package.__path__`处理的一部分
	-	基于路径的查找器调用其处理路径条目，以获取路径条目
		查找器

	> - **导入路径钩子返回路径条目查找器**
	> - `sys.path_hooks`为导入路径钩子列表，可在其中注册
		定制导入路径钩子

###	默认元路径查找器/导入器

python默认实现`sys.meta_path`有以下导入器（元路径查找器）

-	`BuiltinImporter`：定位、导入内置模块
-	`FrozenImporter`：定位、导入冻结模块
-	`PathFinder`：定位、导入来自*import path*中模块

> - 尝试导入模块时，内置模块、冻结模块导入器优先级较高，所以
	解释器首先搜索**内置**模块

##	*Finder*

-	指定名称模块在`sys.modules`找不到时，python继续搜索
	`sys.meta_path`，按顺序调用其中元路径查找器

-	若`sys.meta_path`处理到列表末尾仍未返回说明对象，则
	`raise ModuleNotFoundError`

> - 导入过程中引发的任何异常直接向上传播，并放弃导入过程
> - 对非最高层级模块的导入请求可能会多次遍历元路径

###	*Meta Path Finders*

元路径查找器：

-	元路径查找器可使用任何策略确定其是否能处理给定名称模块
	-	若知道如何处理指定名称的模块，将返回模块规格说明
	-	否则返回`None`

-	模块规格协议：元路径查找器应实现`find_spec()`方法
	-	接受名称、导入路径、目标模块作为参数
	-	返回模块规格说明

###	*Spec*

-	模块规格[说明]：基于每个模块封装的模块导入相关信息
	-	模块规格中大部分信息对所有模块是公用的
	-	模块规格说明作为模块对象的`__spec__`属性对外公开

-	用途
	-	允许状态在导入系统各组件间传递，如：查询器和加载器
	-	允许导入机制执行加载的样板操作，否则该由加载器负责

###	`find_spec`

```python
def finder.find_spec(fullname, path=None, target=None):
	pass
```

-	`fullname`：被导入模块的完整限定名称
-	`path`：供模块搜索使用的路径条目
	-	对最高层级模块应为`None`
	-	对子模块、子包应为父包`__path__`属性值，若
		相应`__path__`属性无法访问将
		`raise ModuleNotFoundError`
-	`target`：将被作为稍后加载目标的现有模块对象
	-	导入系统仅在重加载期间传入目标模块

> - 导入器的`find_spec()`返回模块规格说明中加载器为`self`
> - 有些元路径查找器仅支持顶级导入，`path`参数不为`None`时
	总返回`None`

##	*Loaders*

-	模块规格说明被找到时，导入机制将在加载该模块时使用
	-	其中包含的加载器将被使用，若存在

###	加载流程

```python
module = None

if spec.loader is not None and hasattr(spec.loader, 'create_module'):
	# 模块说明中包含加载器，使用加载器创建模块
	module = spec.loader.create_module(spec)

if module is None:
	# 否则创建空模块
	module = types.ModuleType(spec.name)

 # 设置模块导入相关属性
_init_module_attrs(spec, module)

if spec.loader is None:
	# 模块说明中不包含加载器
	# 检查模块是否为为命名空间包
	if spec.submodule_search_locations is not None:
		# 设置`sys.modules`
		sys.modules[spec.name] = module
	else:
		raise ImportError
elif not hasattr(spec.loader, "exec_module"):
	# 向下兼容现有`load_module`
	module = spec.loader.load_module(spec.name)
else:
	sys.modules[spec.name] = module
	try:
		# 模块执行
		spec.loader.exec_module(module)
	except BaseException:
		try:
			# 加载模块失败则从`sys.modules`中移除
			del sys.modules[spec.name]
		except KeyError:
			pass
		raise
return sys.modules[spec.name]
```

-	创建模块对象
-	设置模块导入相关属性：在执行模块代码前设置
-	`sys.modules`注册模块
-	模块执行：模块导入关键，填充模块命名空间

####	`create_module`创建模块对象

-	模块加载器可以选择通过实现`create_module`方法在加载
	期间创建模块对象
	-	其应接受模块规格说明作为参数

-	否则导入机制使用`types.ModuleType`自行创建模块对象

####	`sys.modules`注册模块

-	在加载器执行代码前注册，避免模块代码导入自身导致无限
	递归、多次加载
-	若模块为命名空间包，直接注册空模块对象

####	`exec_module`模块执行

-	导入机制调用`importlib.abc.Loader.exec_module()`方法执行
	模块对象

	> - CPython：`exec_module`不定返回传入模块，其返回值将被
		忽略
	> > -	`importlib`避免直接使用返回值，而是通过在
			`sys.modules`中查找模块名称获取模块对象
	> > -	可能会间接导致被导入模块可能在`sys.modules`中
			替换其自身

-	加载器应该该满足

	-	若模块是python模块（非内置、非动态加载），加载器应该
		在模块全局命名空间`module.__dict__`中执行模块代码

	-	若加载器无法执行指定模块，则应`raise ImportError`，
		在`exec_module`期间引发的任何其他异常同样被传播

-	加载失败时作为附带影响被成功加载的模块仍然保留

	> - 重新加载模块会保留加载失败模块（最近成功版本）

##	*Path Based Finder*--`PathFinder`

基于路径的查找器：在特定*path entry*中查找、加载指定的python
模块、包

-	基于路径查找器只是遍历*import path*中的路径条目，将其
	关联至处理特定类型路径的*path entry finder*

-	默认路径条目查找器集合实现了在文件系统中查找模块的所有
	语义，可以处理多种文件类型

	-	python源码`.py`
	-	python字节码`.pyc`
	-	共享库`.so`
	-	zip包装的上述文件类型（需要`zipimport`模块支持）

-	作为元路径查找器
	-	实现有`find_spec`协议
	-	并提供额外的钩子、协议以便能扩展、定制可搜索路径条目
		的类型，定制模块从*import path*的查找、加载

###	流程

导入机制调用基于路径的查找器的`find_spec()`迭代搜索
*import path*的路径条目，查找对应路径条目查找器

-	先在`sys.path_impporter_cache`缓存中查找对应路径条目
	查找器

-	若没有在缓存中找到，则迭代调用`sys.path_hooks`中
	*Path Entry Hook*

-	迭代结束后若没有返回路径条目查找器，则

	-	置`sys.path_importer_cache`对应值为`None`
	-	返回`None`，表示此元路径查找器无法找到该模块

####	当前目录

对空字符串表示的当前工作目录同`sys.path`中其他条目处理方式
有所不同

-	若当前工作目录不存在，则`sys.path_importer_cache`
	中不存放任何值

-	模块查找回对当前工作目录进行全新查找

-	`sys.path_importer_cache`使用、
	`importlib.machinery.PathFinder.find_spec()`返回路径将是
	实际当前工作目录而非空字符串

###	*Path Entry Hook*

路径条目钩子：根据路径条目查找对应路径条目查找器的可调用对象

-	参数：字符串、字节串，表示要搜索的目录条目
	-	字节串的编码由钩子自行决定
	-	若钩子无法解码参数，应`raise ImportError`

-	路径条目钩子返回值
	-	可处理路径条目的路径条目查找器
	-	`raise ImportError`：表示钩子无法找到与路径条目对应
		路径条目查找器
		-	该异常会被忽略，并继续对*import path*迭代

###	*Path Entry Finder*--`PathEntryFinder`

路径条目查找器：

> - 元路径查找器作用于导入过程的开始，遍历`sys.meta_path`时
> - 路径条目查找器某种意义上是**基于路径查找器的实现细节**

####	`find_spec`

```python
def PathEntryFinder.find_spec(fullname, target=None):
	pass
```

-	路径条目查找器协议：目录条目查找器需实现`find_spec`方法
	-	以支持模块、已初始化包的导入
	-	给命名空间包提供组成部分

-	参数
	-	`fullname`：要导入模块的完整限定名称
	-	`target`：目标模块

-	返回值：完全填充好的模块规格说明
	-	模块规格说明总是包含加载器集合
	-	但命名空间包的规格说明中`loader`会被设置为`None`，
		并将`submodule_search_locations`设置为包含该部分的
		列表，以告诉导入机制该规格说明为命名空间包的portion

> - *Portion*：构成命名空间包的单个目录内文件集合
> - 替代旧式`find_loader()`、`find_module()`方法

###	替换标准导入系统

-	替换`sys.meta_path`为自定义元路径钩子
	-	替换整个导入系统最可靠机制

-	替换内置`__import__()`函数
	-	仅改变导入语句行为而不影响访问导入系统其他接口
	-	可以在某个模块层级替换，只改变某块内部导入语句行为

-	替换`find_spec()`，引发`ModuleNotFoundError`
	-	选择性的防止在元路径钩子导入某些模块

###	`__main__`

-	`__main__`模块是在解释器启动时直接初始化，类似`sys`、
	`builtins`，但是不被归类为内置模块，因为其初始化的方式
	取决于启动解释器的旗标（命令行参数）

####	`__spec__`

根据`__main__`被初始化的方式，`__main__.__spec__`被设置为
`None`或相应值

-	`-m`选项启动：以脚本方式执行模块

	-	此时`__spec__`被设置为相应模块、包规格说明

	-	`__spec__`会在`__main__`模块作为执行某个目录、zip
		文件、其他`sys.path`条目的一部分加载时被填充

	-	此时`__main__`对应可导入模块和`__main__`被视为不同
		模块

-	其余情况

	-	`__spec__`被设置为`None`

	-	因为用于填充`__main__`的代码不直接与可导入模块相对应
		-	交互型提示
		-	`-c`选项
		-	从stdin运行
		-	从源码、字节码文件运行

> - `-m`执行模块时`sys.path`首个值为空字符串，而直接执行脚本
	时首个值为脚本所在目录

##	*Import[ Search] Path*定制

###	动态增加路径

```python
import sys
sys.path.insert(1, /path/to/fold/contains/module)
	# 临时生效，对不经常使用的模块较好
```

###	修改`PYTHONPATH`环境变量

```shell
 # .bashrc
export PYTHONPATH=$PYTHONPATH:/path/to/fold/contains/module
```

-	对许多程序都使用的模块可以采取此方式
-	会改变所有Python应用的搜索路径

###	增加`.pth`文件

在`/path/to/python/site-packages`（或其他查找路径目录）下
添加`.pth`配置文件，内容为需要添加的路径

```conf
 # extras.pth
/path/to/fold/contains/module
```

-	简单、推荐
-	python在遍历已知库文件目录过程中，遇到`.pth`文件会将其中
	路径加入`sys.path`中

