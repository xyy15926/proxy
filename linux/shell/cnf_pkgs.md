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
updated: 2021-08-30 21:37:20
toc: true
mathjax: true
description: 
---

##	库、依赖

###	`ldd`、`/lib/ld-linux.so`

-	`/lib/ld-linux.so`：共享可执行库帮助文件
-	`ldd`：查看程序所需共享库的 *Bash* 脚本
	-	通过调用 `/lib/ld-linux.so`、设置环境变量完成功能
		-	`LD_TRACE_LOADED_OBJECTS`：非空时，可执行程序执行只显示模块的依赖，不真正执行
		-	`LD_WARN`
		-	`LD_BIND_NOW`
		-	`LD_LIBRARY_VERSION`
		-	`LD_VERBOSE`

		```sh
		$ /lib/ld-linux.so* --list exe
		$ ldd exe
			// 二者等价
		```

-	选项参数
	-	`-v`：详细模式
	-	`-u`：打印未使用的直接依赖
	-	`-d`：执行重定位，报告任何丢失对象
	-	`-r`：执行数据、函数重定位，报告任何丢失的对象、函数

-	输出说明
	-	第1列：程序动态链接库依赖
	-	第2列：系统提供的与程序需要的库所对应的库
	-	第3列：库加载的开始地址
	-	首行、尾行可能是两个由 *kernel* 向所有进程都注入的库

###	`ldconfig`

-	`ldconfig`：创建、查看动态链接库缓存
	-	根据 `/etc/ld.so.conf` 文件中包含路径
		-	搜索动态链接库文件
		-	创建缓存文件 `/etc/ld.so.cache`
	-	动态链接库默认包含 `/lib`、`/lib64`、`/usr/lib`、`/usr/lib64`
		-	优先级逐渐降低，且低于 `/etc/ld.so.conf` 中路径

-	选项参数
	-	生成动态链接库缓存，并打印至标准输出
	-	`-v/--verbose`：详细版
	-	`-n`：仅扫描命令行指定目录
	-	`-N`：不更新缓存
	-	`-X`：不更新文件链接
	-	`-p`：查看当前缓存中库文件
	-	`-f <CONF>`：指定配置文件，默认 `/etc/ld.so.conf`
	-	`-C <CACHE>`：指定生成缓存文件
	-	`-r <ROOT>`：指定执行根目录，默认 `/`（调用 `chroot` 实现）
	-	`-l`：专家模式，手动设置
	-	`-f <Format>/--format=<Format>`：缓存文件格式
		-	`ld`：老格式
		-	`new`：新格式
		-	`compat`：兼容格式

##	命令总结

-	库、依赖查看
	-	`ldconfig`：创建、查看动态链接库缓存
	-	`ldd`：查看程序所需共享库的 *Bash* 脚本
	-	`strings`：查看系统 *glibc* 支持的版本
	-	`objdump`：查看目标文件的动态符号引用表

-	包管理器
	-	`apt`
	-	`rpm`
	-	`yum`
	-	`dpkg`
	-	`zypper`
	-	`pacman`




