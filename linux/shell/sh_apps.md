---
title: Shell 应用程序
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Package Manager
  - Dependency
date: 2021-07-29 21:55:00
updated: 2021-07-29 21:55:57
toc: true
mathjax: true
description: 
---

##	包管理

###	`apt`

-	`install`
-	`update`
-	`remove`
-	`autoremove`
-	`clean`

###	`rpm`

###	`yum`

####	说明

-	`yum`在不使用`yum makecache`手动检查源配置文件时，可能
	很长时间都不会更新cache，也就无法发现软件仓库的更新
	（当然，可能和仓库类型有关，使用ISO镜像作为仓库时如此，
	即使不挂载镜像，yum在执行其他操作也不会发现有个仓库无法
	连接）

###	`dpkg`

###	`zypper`

###	`pacman`

##	库、依赖

###	`ldconfig`

创建、查看动态链接库缓存

-	根据`/etc/ld.so.conf`文件中包含路径，搜索动态链接库文件
	，创建缓存文件`/etc/ld.so.cache`

-	默认包含`/lib`、`/lib64`、`/usr/lib`、`/usr/lib64`，
	优先级逐渐降低，且低于`/etc/ld.so.conf`中路径

####	参数

-	生成动态链接库缓存，并打印至标准输出
-	`-v/--verbose`：详细版
-	`-n`：仅扫描命令行指定目录
-	`-N`：不更新缓存
-	`-X`：不更新文件链接
-	`-p`：查看当前缓存中库文件
-	`-f CONF`：指定配置文件，默认`/etc/ld.so.conf`
-	`-C CACHE`：指定生成缓存文件
-	`-r ROOT`：指定执行根目录，默认`/`（调用`chroot`实现）
-	`-l`：专家模式，手动设置
-	`-f Format/--format=Format`：缓存文件格式
	-	`ld`：老格式
	-	`new`：新格式
	-	`compat`：兼容格式

###	`ldd`

查看程序所需共享库的bash脚本

-	通过设置一系列环境变量，如`LD_TRACE_LOADED_OBJECTS`、
	`LD_WARN`、`LD_BIND_NOW`、`LD_LIBRARY_VERSION`、
	`LD_VERBOSE`等

-	当`LD_TRACE_LOAD_OBJECTS`环境变量不空时，任何可执行程序
	运行时只显示模块的依赖，且程序不真正执行

-	实质上是通过`ld-linux.so`实现

	```c
	$ /lib/ld-linux.so* --list exe
	$ ldd exe
		// 二者等价
	```

	> - `ld-linux.so*`参见`cppc/func_lib.md`

####	参数

-	`-v`：详细模式
-	`-u`：打印未使用的直接依赖
-	`-d`：执行重定位，报告任何丢失对象
-	`-r`：执行数据、函数重定位，报告任何丢失的对象、函数

####	打印

-	第1列：程序动态链接库依赖
-	第2列：系统提供的与程序需要的库所对应的库
-	第3列：库加载的开始地址

> - 首行、尾行可能是两个由kernel向所有进程都注入的库

###	`strings`

查看系统*glibc*支持的版本

###	`objdump`

查看目标文件的动态符号引用表
