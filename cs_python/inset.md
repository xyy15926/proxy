---
title: Python安装配置
tags:
  - Python
categories:
  - Python
date: 2019-03-21 17:27:15
updated: 2019-02-28 15:12:54
toc: true
mathjax: true
comments: true
description: Python安装配置
---

##	Python

###	Python3包管理安装

-	CentOS7依赖缺失
	-	`zlib-devel`
	-	`bzip2-devel`
	-	`readline-devel`
	-	`openssl-devel`
	-	`sqlite(3)-devel`

> - 根据包用途、名称可以确认对应的应用，缺少就是相应
	`-devel(-dev)`

###	Python Implementation

名称中flag体现该发行版中python特性

-	`-d`：with pydebug

-	`-m`：with pymalloc

-	`-u`：with wide unicode

> - pymalloc是specialized object allocator
> > -	比系统自带的allocator快，且对python项目典型内存
		分配模式有更少的开销
> > -	使用c的`malloc`函数获得更大的内存池
> - 原文：Pymalloc, a specialized object allocator written
	by Vladimir Marangozov, was a feature added to Python2.1.
	Pymalloc is intended to be faster than the system
	malloc() and to have less memory overhead for allocation
	patterns typical of Python programs. The allocator uses
	C's malloc() function to get large pools of memory and
	then fulfills smaller memory requests from these pools.J

>	注意：有时也有可能只是hard link

##	Python配置

###	Python相关环境变量

-	`PYTHONPATH`：python库查找路径
-	`PYTHONSTARTUP`：python自动执行脚本

###	自动补全

```python
import readline
import rlcompleter
	# 为自动补全`rlcompleter`不能省略
import atexit
readline.parse_and_bind("tab:complete")
	# 绑定`<tab>`为自动补全

try:
	readline.read_history("/path/to/python_history")
		# 读取上次存储的python历史
except:
	pass
atexit.register(
	readline.write_history_file,
	"/path/to/python_history"
)
	# 将函数注册为推出python环境时执行
	# 将python历史输入存储在的自定以文件中
	# 这部分存储、读取历史其实不必要

del readline, rlcompleter
```

-	**每次**在python解释器中**执行**生效

-	保存为文件`python_startup.py`，将添加到环境变量
	`PYTHONSTARTUP`中，每次开启python自动执行
	```shell
	# .bashrc
	export PYTHONSTARTUP=pythonstartup.py
		# 这个不能像*PATH一样添加多个文件，只能由一个文件
	```

##	Pip

python包、依赖管理工具

-	pip包都是源码包 

	-	需要在安装时编译，因此可能在安装时因为系统原因出错
	-	现在有了`wheels`也可以安装二进制包

###	安装

-	编译安装python一般包括`pip`、`setuptools`
-	系统自带python无`pip`时，可用`apt`、`yum`等工具可以直接安装
-	虚拟python环境，无般法使用系统包管理工具安装pip，则只能
	下载`pip`包使用`setuptools`安装

###	配置

配置文件：`~/.config/pip/pip.conf`

```cnf
[global]
index-url = https:?//pypi.tuna.tsinghua.edu.cn/simple/
	# pypi源地址
format = columns
	# pip list输出格式（legacy，columns）
```

###	依赖管理

pip通过纯文本文件（一般命名为`requirements.txt`）来记录、
管理python项目依赖

-	`$ pip freeze`：按照`package_name=version`的格式输出
	已安装包
	`$ pip install -r`：可按照指定文件（默认`requirements.txt`）
	安装依赖

## Virtualenv/Venv

虚拟python环境管理器，使用`pip`直接安装

-	将多个项目的python依赖隔离开，避免多个项目的包依赖、
	python版本冲突
-	包依赖可以安装在项目处，避免需要全局安装python包的权限
	要求、影响

###	实现原理

`$ virtualenv venv-dir`复制python至创建虚拟环境的文件夹中，
`$ source venv-dir/bin/activate`即激活虚拟环境，修改系统环境
变量，把python、相关的python包指向当前虚拟环境夹

###	Virtualenv使用

##	Pyenv

python版本管理器，包括各种python发行版

###	安装

不需要事先安装python

-	从github获取pyenv：<git://github.com/yyuu/pyenv.git>

-	将以下配置写入用户配置文件（建议是`.bashrc`)，也可以在
	shell里面直接执行以暂时使用

	```shell
	export PYENV_ROOT="$HOME/pyenv路径"
	export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init -)"
	```

	以上配置可参见`home_config/bashrc_addon`，以项目详情为准

###	Pyenv安装Python发行版问题
	
使用pyenv安装python时一般是从`PYTHON_BUILD_MIRROR_URL`表示
的地址下载安装文件（或者说整个系统都是从这个地址下载），缺省
是<http://pypi.python.org>，但国内很慢
	#todo

-	设置这个环境变量为国内的镜像站，如
	<http://mirrors.sohu.com/python>，但这个好像没用
-	在镜像站点下载安装包，放在`pyenv/cache`文件夹下（没有就
	新建）

pyenv安装python时和使用一般安装应用一样，只是安装prefix不是
`/usr/bin/`之类的地方，而是pyenv安装目录，因此pyenv编译安装
python也需要先安装依赖

###	实现原理

####	修改`$PATH`环境变量
		
-	用户配置文件将`PYENV_ROOT/bin`放至`$PATH`首位
-	初始化pyenv时会将`PYENV_ROOT/shims`放至$PATH首位

`shims`、`bin`放在最前，优先使用pyenv中安装的命令

-	`bin`中包含的是pyenv自身命令（还有其他可执行文件，但是
	无法直接执行?）

-	`shims`则包含的是**所有**已经安装python组件

	-	包括python、可以执行python包、无法直接执行的python包

	-	这些组件是内容相同的脚本文件，仅名称是pyenv所有安装
		的python包

		-	用于截取python相关的命令
		-	并根据设置python发行版做出相应的反应
		-	因此命令行调用安装过的python包，pyenv会给提示
			即使不是安装在当前python环境中

因此一般将命令放在`.profile`文件中，这样每次登陆都会设置好
pyenv放在`.bashrc`中会设置两次（没有太大关系）

####	使用指定Python发行版

-	`$ pyenv local py-version`指定是在文件夹下生成
	`.python-version`文件，写明python版本

-	所有的python相关的命令都被shims中的脚本文件截取

pyenv应该是逐层向上查找`.python-version`文件，查找到文件则
按照相应的python发行版进行执行，否则按global版本

##	Conda

通用包管理器

-	管理任何语言、类型的软件

	-	conda默认可以从<http://repo.continuum.io>安装**已经
		编译好**二进制包

	-	conda包和pip包只是**部分**重合，有些已安装conda包
		甚至无法被pip侦测到（非python脚本包）

	-	python本身也作为conda包被管理

-	创建、管理虚拟python环境（包括python版本）

###	安装

-	conda在Miniconda，Anaconda发行版中默认安装

	-	Miniconda是只包括conda的发行版，没有Anaconda中默认
		包含的包丰富

	-	在其他的发行版中可以直接使用pip安装，但是这样安装的
		conda功能不全，可能无法管理包

-	Miniconda、Anaconda安装可以自行设置安装位置，无需介怀

###	配置

conda配置文件为`$HOME/.condarc`，其中可以设置包括源在内
	配置

```cnf
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

####	添加国内源

conda源和pypi源不同（以下为清华源配置，当然可以直接修改
配置文件）

```shell
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
$ conda config --set show_channel_urls yes
```

> - conda源不是pypi源，不能混用

####	Win平台设置

-	添加菜单项

	```sh
	 # 可用于恢复菜单项
	$ cd /path/to/conda_root
	$ python .\Lib\_nsis.py mkmenus
	```

-	VSCode是通过查找、执行`activate.bat`激活虚拟环境
	-	所以若虚拟环境中未安装`conda`（无`activate.bat`）
		则虚拟环境无法自动激活

###	常用命令

```shell
$ conda create [--clone ori_env] -n env [packages[packages]]
	# 创建虚拟环境
	# python也是conda包，可指定`python=x.x`
$ conda remove -n env --all
	# 删除虚拟环境

$ conda info -e
$ conda env list
	# 列出虚拟环境
$ conda info
	# 列出conda配置

$ conda activate env
	# 激活env环境
$ conda deactivate
	# 退出当前环境

$ conda list -n env
	# 列出env环境/当前环境安装conda包
$ conda search package
	# 搜索包
$ conda install [-n env] packages
	# env环境/当前环境安装conda包
$ conda update [-n env] packages
	# env环境/当前环境升级conda包
$ conda remove [-n env] packages
	# env环境/当前环境移除包
```

> - 使用conda而不是pip安装包更合适，方便管理

-	创建新环境时，默认不安装任何包，包括`pip`，此时切换到
	虚拟环境后，`pip`等命令都是默认环境的命令

##	Pipenv

pip、virtualenv、Pipfile（版本控制）功能的综合，事实上就
依赖于pip、virtualenv（功能封装）

-	`$ pipenv sync/install`替代`$ pip install`
-	`$ pipenv shell`替代`$ activate`
-	`$ pipenv run`甚至可以不用激活虚拟环境运行某些命令
-	`Pipfile`控制dev、release包管理，`Pipfile.lock`锁定包依赖

###	安装

-	使用pip直接安装
-	系统安装源里有pipenv，也可以用系统包管理工具安装

###	实现原理

####	Python版本

pipenv和virtualenv一样指定python版本也需要已经安装该python
版本

-	`$PATH`中的路径无法寻找到相应的python版本就需要手动
	指定

-	不是有版本转换，将当前已安装版本通过类似2to3的“中
	间层”转换为目标版本

####	虚拟环境

pipenv会在`~/.local/share/virtualenv`文件夹下为所有的虚拟
python环境生成文件夹

-	文件夹名字应该是“虚拟环境文件夹名称-文件夹全路径hash”

-	包括已安装的python包和python解释器

-	结构和virtualenv的内容类似，但virtualenv是放在项目目录下

-	`$ python shell`启动虚拟环境就是以上文件夹路径放在
	`$PATH`最前

####	依赖管理

pipenv通过Pipfile管理依赖（环境）

-	默认安装：`$ pipenv install pkg`

	-	作为默认包依赖安装pkg，并记录于`Pipfile`文件
		`[packages]`条目下

	-	相应的`$ pipenv install`则会根据`Pipfile`文件
		`[packages]`条目安装默认环境包依赖

-	开发环境安装：`$ pipenv install --dev pkg`

	-	作为开发环境包依赖安装pkg，并记录于`Pipfile`
		文件`[dev-packages]`条目下

	-	相应的`$ pipenv intall --dev`则会根据`Pipfile`
		文件中`[dev-packages]`安装开发环境包依赖

####	Pipfile和Pipfile.lock

-	`Pipfile`中是包依赖**可用（install时用户指定）**版本
-	`Pipfile.lock`则是包依赖**具体**版本
	-	是pipenv安装包依赖时具体安装的版本，由安装时包源的
		决定
	-	`Pipfile.lock`中甚至还有存储包的hash值保证版本一致
	-	`Pipfile`是用户要求，`Pipfile.lock`是实际情况

因此

-	`$ pipenv install/sync`优先依照`Pipfile.lock`安装具体
	版本包，即使有更新版本的包也满足`Pipfile`的要求

-	`Pipfile`和`Pipfile.lock`是同时更新、内容“相同”，
	而不是手动锁定且手动更新`Pipfile`，再安装包时会默认更新
	`Pipfile.lock`

### Pipenv用法

详情<https://docs.pipenv.org>

####	创建新环境

具体查看`$pipenv --help`，只是记住`$pipenv --site-packages`
表示虚拟环境可以共享系统python包

####	默认环境和开发环境切换

pipenv没有像git那样的切换功能

-	默认环境“切换”为dev环境：`$ pipenv install --dev`
-	dev环境“切换”为默认环境：`$ pipenv uninstall --all-dev`

####	同步

`$ pipenv sync`

官方是说从`Pipfile.lock`读取包依赖安装，但是手动修改`Pipfile`
后`$ pipenv sync`也会先更新`Pipfile.lock`，然后安装包依赖，
感觉上和`$ pipenv install`差不多

###	Pipenv特性

####	和Pyenv的配合

pipenv可以找到pyenv已安装的python发行版，且不是通过`$PATH`
中`shims`获得实际路径

-	pipenv能够找到pyenv实际安装python发行版的路径versions，
	而不是脚本目录`shims`

-	pipenv能自行找到pyenv安装的python发行版，即使其当时没有
	被设置为local或global

	-	pyenv已安装Anaconda3和3.6并指定local为3.6的情况下
		`$ pipenv --three`生成的虚拟python使用Anaconda3

	-	后系统全局安装python34，无local下`pipenv --three`
		仍然是使用Aanconda3

	-	后注释pyenv的初始化命令重新登陆，`pipenv --three`就
		使用python34

目前还是无法确定pipenv如何选择python解释器，但是根据以上测试
和github上的feature介绍可以确定的是和pyenv命令有关

pipenv和pyenv一起使用可以出现一些很蠢的用法，比如：pyenv指定
的local发行版中安装pipenv，然后用这pipenv可以将目录设置为
另外版本虚拟python环境（已经系统安装或者是pyenv安装）

##	总结

除了以上的包管理、配置工具，系统包管理工具也可以看作是python
的包管理工具

-	事实上conda就可以看作是pip和系统包管理工具的交集
-	系统python初始没有pip一般通过系统包管理工具安装

###	使用场景

优先级：pip > conda > 系统包管理工具

-	纯python库优先使用pip安装，需要额外编译的库使用conda
-	conda源和系统源都有的二进制包，优先conda，版本比较新

####	2018/04/06经验

最后最合适的多版本管理是安装pipenv

-	系统一般自带python2.7，所以用系统包管理工具安装一个
	python3

-	使用新安装的python3安装pipenv，因为系统自带的python2.7
	安装的很多包版过低

-	最后如果对python版本要求非常严格
	-	还可以再使用pyenv安装其他版本
	-	然后仅手动启用pyenv用于指示pipenv使用目标python版本

####	2019/02/20经验

直接全局（如`/opt/miniconda`）安装Miniconda也是很好的选择

-	由conda管理虚拟环境，虚拟环境创建在用户目录下，登陆时
	激活


