---
title: GCC
categories:
  - Linux
tags:
  - Tool
  - Compiler
  - GCC
  - G++
date: 2019-05-20 22:27:04
updated: 2021-07-16 16:57:32
toc: true
mathjax: true
comments: true
description: GCC
---

##	G++

`g++`：是`gcc`的特殊版本，链接时其将自动使用C++标准库而不是
C标准库

```c
$ gcc src.cpp -l stdc++ -o a.out
	// 用`gcc`编译cpp是可行的
```

