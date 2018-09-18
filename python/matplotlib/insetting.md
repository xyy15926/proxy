#	Matplotlib配置

##	配置文件`matplotlibrc`

###	配置文件路径

-	`/current/dir/matplotplibrc`
-	`$MATPLOTLIBRC/matplotlibrc`
-	`~/.config/matplotlib/matplotlibrc(~/.matplotlib/matplotlibrc)`
-	`/path/to/matlabplot/mpl-data/matplotlibrc`

###	查看实际路径

`python -c 'import matplotlib as mpl; print(mpl.matplotlib_fname())'`

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

-	mpl不兼容ttc文件，只能识别ttf文件（ttc和ttf文件可查阅其他笔记）
-	mpl默认使用`/path/to/matplotlib/mpl-data/fonts`中的字体文件

####	管理单个图表

通过mpl*字体管理器*管理**单个**图表元素字体，对每个需要使用
该字体的图表元素，都需要传递相应的参数

```python
ch=mpl.font_manager.FontProperties(fname=/path/to/font.ttf)
ax.set_title(name, fontproperties=ch)
```

优势：连字体都不需要安装

####	运行时配置MPL默认字体

运行时指定"已安装"字体给**当前**mpl配置中默认字体

```python
mpl.rcParams['font.default-font-family'].insert(0, font-name)
	// 必须在`plt.plot()`类似的绘图语句执行之前执行
```

-	这里的font-name不是ttf文件名，而是字体安装之后的字体
	名称，系统已安装字体可以使用`fc-list`命令查看

-	`mpl.font_manager.FontManager`能找到的字体

	-	系统中已安装ttf中文字体，即
		-	`/usr/share/fonts/`中ttf文件对应字体
		-	?`~/.local/share/fonts/`中ttf文件对应字体

	-	`/path/to/matplotlib/mpl-data/fonts/`中字体文件对应
		字体

	-	使用以下命令得到mpl能够找到、使用的字体名称
		```python
		sorted[i.name for i in mpl.font_manager.FontManger.ttflist]
		```

-	不是所有系统已安装字体，mpl均可找到、使用
	-	字体文件问题，如前所述不兼容ttc字体文件
	-	缓存问题，mpl可能在`~/.cache/matplotlib/`下有cache
		文件
		-	mpl会从里面的一个json文件读取可用字体，需删
		-	可以使用以下命令得到cache文件夹
			```python
			mpl.get_cachedir()
			```
	-	即使`mpl.font_manager.FontManager.ttflist`中包含的
		字体，mpl也可能不能直接使用，只有json文件中有记录的
		字体才能使用（json文件包含有个字体对应的文件全名），
		因此，安装新字体之后需要删除json文件

-	查看当前默认font-family
	```python
	mpl.font_manager.FontProperties().get_family()
	```
	一般默认字体族是"sans-serif"，所以应该将字体名称插入其首位
	```python
	mpl.rcParams['font.sans-serif'].insert(0, font-name)
	```

####	修改MPL配置文件

修改配置文件更改**全局**默认字体

-	将字体名称添加在默认字体族首位，作为最优先候选字体
	```
	font.sans-serif : font-name,
	```

-	同样要求mpl能够找到、使用相应的字体文件

	#todo:axes.unicode_minus:False，#作用就是解决负号'-'显示为方块的问题

##	图片显示

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

```python
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
	// `mpl.use("Agg")` must be put before this
```

