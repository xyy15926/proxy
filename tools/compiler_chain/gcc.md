---
title: GCC
tags:
  - 工具
  - 编译工具
categories:
  - 工具
  - 编译工具
date: 2019-05-20 22:27:04
updated: 2019-05-20 22:27:04
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

