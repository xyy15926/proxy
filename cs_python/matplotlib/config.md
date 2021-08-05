---
title: Matplotlib 配置
categories:
  - Python
  - Matplotlib
tags:
  - Python
  - Matplotlib
  - Data Visualization
date: 2019-03-21 17:27:37
updated: 2020-09-09 13:17:38
toc: true
mathjax: true
comments: true
description: Matplotlib配置
---

##	配置文件`matplotlibrc`

###	配置文件路径

-	`/current/dir/matplotplibrc`
-	`$MATPLOTLIBRC/matplotlibrc`
-	`~/.config/matplotlib/matplotlibrc(~/.matplotlib/matplotlibrc)`
-	`/path/to/matlabplot/mpl-data/matplotlibrc`

###	查看实际路径

```shell
$ python -c 'import matplotlib as mpl; print(mpl.matplotlib_fname())'
```

##	中文问题

###	字符编码问题

-	python2中默认字符编码方案是ascii，需要更改为utf-8

	```python
	import sys
	reload(sys)
	sys.setdefaultencoding("utf8")
	```

-	python3则默认为utf-8编码，无需其他操作

###	字体文件缺失

-	mpl不兼容ttc文件，只能识别ttf文件
-	mpl默认使用`/path/to/matplotlib/mpl-data/fonts`中的字体
	文件

> - ttc和ttf格式参见*cs_program/character*

####	管理单个图表

-	通过mpl*字体管理器*管理**单个**图表元素字体
	-	对每个需要使用该字体的图表元素，都需要传递相应的参数
	-	无需字体都不需要安装

```python
ch=mpl.font_manager.FontProperties(fname=/path/to/font.ttf)
ax.set_title(name, fontproperties=ch)
```

####	运行时配置MPL默认字体


```python
 # 必须在`plt.plot()`类似的绘图语句执行之前执行
 # `font-name`为已安装字体名称而不是字体文件名
mpl.rcParams['font.default-font-family'].insert(0, font-name)
 # 查看当前默认font-family
mpl.font_manager.FontProperties().get_family()
 # 一般默认字体族是"sans-serif"，所以应该将字体名称插入其首位
mpl.rcParams['font.sans-serif'].insert(0, font-name)
```

-	运行时指定“已安装”字体给**当前执行**mpl配置中默认字体

	-	系统中已安装**ttf字体**，即`fc-`查找目录
		-	`/usr/share/fonts/`
		-	`$HOME/.local/share/fonts/`
		-	`$HOME/.fonts/`

	-	`/path/to/matplotlib/mpl-data/fonts/`中字体文件对应
		字体


-	已安装字体指`mpl.font_manager.FontManager`能找到字体，但
	不是所有系统已安装字体，mpl均可找到、使用

	```python
	# 获取mpl所有可用字体并排序
	sorted([i.name for i in mpl.font_manager.FontManger().ttflist])
	```

	-	字体文件问题：mpl不兼容ttc字体文件

	-	缓存问题：mpl可能在`~/.cache/matplotlib/`下有cache
		文件，mpl会直接从中获取可用字体列表，新字体可能需
		删除才能在mpl中生效

		```python
		# 获取mpl缓存文件夹
		mpl.get_cachedir()
		```

####	修改MPL配置文件

-	修改配置文件更改**全局**默认字体

	-	将字体名称添加在默认字体族首位，作为最优先候选字体

		```cnf
		font.sans-serif : font-name,
		```

-	同样要求mpl能够找到、使用相应的字体文件

##	图片显示

###	X11依赖

-	mpl一般在常用于包含有X11的图形化界面中，所以若X11不可用
	则需修改mpl配置

-	对生成pdf、png、svg等，可以修改后端为`Agg`

	-	运行时修改

		```python
		import matplotlib as mpl
		# 须在导入`pyplot`前执行
		mpl.use("Agg")
		from matplotlib import pyplot as plt
		```

	-	修改配置文件

		```cnf 
		backend: TkAgg
		```

>	By default, matplotlib ships configured to work with a
	graphical user interface which may require an X11
	connection. Since many barebones application servers
	don't have X11 enabled, you may get errors if you don't
	configure matplotlib for use in these environments.
	Most importantly, you need to decide what kinds of
	images you want to generate(PNG, PDF, SVG) and configure
	the appropriate default backend. For 99% of users, this
	will be the Agg backend, which uses the C++ antigrain
	rendering engine to make nice PNGs

