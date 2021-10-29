---
title: GCC
categories:
  - Linux
  - Tool
tags:
  - Linux
  - Tool
  - Compiler
  - GCC
  - G++
date: 2019-05-20 22:27:04
updated: 2021-08-30 21:34:50
toc: true
mathjax: true
comments: true
description: GCC
---

##	G++

-	`g++`：`gcc` 的特殊版本，链接时其将自动使用 *C++* 标准库而不是 *C* 标准库 

	```c
	$ gcc src.cpp -l stdc++ -o a.out
		// 用`gcc`编译cpp是可行的
	```

###	相关环境变量

-	`LIBRARY_PATH`：程序编译时，动态链接库查找路径
	-	也可在编译时使用`-R<path>`指定

-	`LD_LIBRARAY_PATH`：程序加载/运行时，动态链接库查找路径
	-	动态链接库寻找由 `/lib/ld-linux.so` 实现，所以可修改其配置达到目的



