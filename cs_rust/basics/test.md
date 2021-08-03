---
title: Rust 测试
categories:
  - Rust
tags:
  - Rust
  - Test
date: 2019-03-21 17:27:37
updated: 2021-08-02 11:09:57
toc: true
mathjax: true
comments: true
description: Rust测试笔记
---

##	测试常用宏、属性注解

###	宏

-	`assert!(exp,...)`
-	`assert_eq!(val1, val2,...)`
-	`assert_ne!(val1, val2,...)`
以上宏均可以传入自定义信息，所有的`...`中参数都将传递给
`format!`宏

###	属性注解

-	`#[test]`：`$>cargo test`时将测试此函数 

-	`#[ignore]`：除非指定，否则`$>cargo test`默认不测试此
	函数（仍需和`#[test]`注解）

-	`#[should_panic(expected=&str)]`：测试中出现`panic`测试
	通过，可以传递`expected`参数，当参数为`panic`信息的起始
	子串才通过

##	`cargo test`命令

命令行参数和可执行文件参数用“--”分隔

```sh
$>cargo test cargo_params -- bin_params 
```

###	可执行文件常用参数

-	控制测试线程数目
```sh
$>cargo test -- --test-thread=1（不使用任何并行机制）
```
-	禁止捕获输出（测试函数中的标准输出）
```sh
$>cargo test -- --nocapture
```
-	测试`#[ignore]`标注的测试函数
```sh
$>cargo test -- --ignore
```

###	命令行常用参数

-	指定部分测试函数
```sh
$>cargo test function_name（cargo匹配以此开头的函数）
```
-	指定部分集成测试文件
```sh
$>cargo test --test test_filename
```

##	单元测试、集成测试

###	单元测试

在隔离环境中一次测试一个模块，可以测试私有接口，常用做法是
在每个文件中创建包含测试函数的tests模块，并使用
`#[cfg(test)]`标注，告诉rust仅在`cargo test`时才编译该mod

###	集成测试

相当于外部库，和用户使用代码的方式相同，只能测试公有接口，
可以同时测试多个模块

新建`/project/tests`目录（和`src`同级），cargo自动寻找此目录
中集成测试文件

-	cargo将每个**文件**当作单独的**crate**编译（模仿用户）
	其中的**文件**也不能共享相同的行为（fn、mod）

	-	需要像外部用户一样`extern crate`引入外部文件，因此
		如果二进制库没有`lib.rs`文件，无法集成测试，推荐
		采用`main.rs`调用`lib.rs`的逻辑结构

	-	不需要添加任何`#[cfg(test)]`注解，cargo会自动将
		`tests`中文件只在cargo test时编译

	-	即使文件中不存在任何`#[test]`注解的测试函数，仍然会
		对其进行测试，只是结果永远是通过

-	而文件夹则不会当作**测试crate**编译

	-	`cargo test`不会将文件夹视为**测试crate**，而是看作
		一个mod

	-	所以可以创建`tests/common/mod.rs`，并在测试文件中
		通过`mod common;`声明定义`common mod`共享行为
		（相当于所有的测试crate = 测试文件 + `common mod`）

