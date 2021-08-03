---
title: 文件、目录访问
categories:
  - Python
  - Py3std
tags:
  - Python
  - Py3std
  - Platform
  - File System
date: 2019-06-09 23:58:05
updated: 2021-08-02 11:50:46
toc: true
mathjax: true
comments: true
description: 文件、目录访问
---

##	`pathlib`

##	`os.path`

###	判断存在

```python
os.path.isdir(r"C:\Users")
os.path.isfile(r"C:\Users")
	# 判断路径名是简单文件、目录
os.path.exists(r"C:\Users")
	# 判断路径名是否存在
```

> - `os.stat`配合`stat`模块有更丰富的功能

###	路径操作

```python
pfile = os.path.join(r"C:\temp", "output.txt")
	# 连接文件名、目录

os.path.split(pfile)
	# 分离文件名、目录

os.path.dirname(pfile)
	# 返回路径中目录
os.path.basename(pfile)
	# 返回路径中
os.path.splitext(pfile)
	# 返回文件扩展名

os.path.normpath(r"C:\temp/index.html")
	# 调整路径为当前平台标准，尤其是分隔符混用时
os.path.abspath("index.html")
	# 返回文件的**完整绝对路径名**
	# 扩展`.`、`..`等语法
```

> - `os.sep`配合字符串`.join`、`.split`方法可以实现基本相同
	效果

##	`fileinput`

##	`stat`

`stat`：包含`os.stat`信息相关常量、函数以便**跨平台**使用

```python
import stat

info = os.stat(filename)
info[stat.ST_MODE]
	# `stat.ST_MODE`就是字符串
	# 只是这样封装易于跨平台
stat.S_ISDIR(info.st_mode)
	# 通过整数`info.st_mode`判断是否是目录
```

> - `os.path`中包含常用部分相同功能函数

##	`glob`

###	`glob.glob`

```python
import glob

def glob.glob(pathname,*,recursive=False)
```

-	参数
	-	`pathname`：文件名模式
		-	接受shell常用文件名模式语法
			-	`?`：单个字符
			-	`*`：任意字符
			-	`[]`：字符选集
		-	`.`开头路径不被以上`?`、`*`匹配
	-	`recursive`
		-	`False`：默认
		-	`True`：`**`将递归匹配所有子目录、文件

-	返回值：匹配文件名列表
	-	目录前缀层次同参数

> - `glob.glob`是利用`glob.fnmatch`模块匹配名称模式

##	`shutil`

`shutil`模块：包含文件操作相关

##	`fnmatch`

##	`linecache`

##	`macpath`

##	`filecmp`

##	`tempfile`

