---
title: Rust 程序设计笔记
categories:
  - Rust
tags:
  - Rust
  - CMD
date: 2019-03-21 17:27:15
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: rust程序设计笔记
---

##	参数、环境变量

-	`std::env::args()`：返回所有参数的一个迭代器，第一参数
	是可执行文件，包含任何无效`unicode`字符将`panic!`，

		args: Vec<String> = std::env::args().collect()

-	`std::env::var("env_var")`：获取设置的环境变量值，返回
	一个Result
	-	环境变量设置时返回包含其的Ok成员
	-	未设置时返回Err成员

	设置“环境变量”并执行程序：`$>ENV_NAME=val cargo run`
