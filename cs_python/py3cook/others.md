---
title: Python
categories:
  - Python
  - Cookbook
tags:
  - Python
  - Cookbook
date: 2019-03-30 09:14:21
updated: 2019-03-30 09:14:21
toc: true
mathjax: true
comments: true
description: Python
---

##	`contextlib`

###	`contextlib.contextmanager`

-	用途：上下文实现装饰器
	-	实现`try...finally...`语句的生成器上下文管理器语法
	-	`try`部分：生成器部分，`with`语句进入时执行
	-	`finally`部分：清理部分，`with`语句退出时执行

-	用法

	-	定义

		```python
		@contextmanager
		def some_generator(<parameters>):
			<setup>
			try:
				yield <value>
			finally:
				<cleanup>
		```

	-	用法

		```python
		with some_generator(<argrument>) as <variable>:
			<body>
		```

	-	等价于

		```python
		<setup>
		try:
			<variable> = <value>
			<body>
		finally:
			<cleanup>
		```









