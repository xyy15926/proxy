---
title: Rust错误（Panic）处理规范
tags:
  - Rust
categories:
  - Rust
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: Rust错误（Panic）处理规范
---

##	panic!与不可预期（不可恢复）错误 

`panic!`时程序默认开始展开（unwinding）、回溯栈并清理函数据

如果希望二进制文件尽量小，可以选择“终止（abort）”，此时
程序内存由操作系统进行清理，在`Cargo.toml`中添加

	[profile]
	panic='abort'

	[profile.release]
	panic='abort'

前者是配置debug时，后者配置release版本

##	Result与潜在（可预期、可恢复）错误

###	Result枚举类型

	Result<T, E>{
		Ok<T>,
		Err<E>,
	}

-	`T`：成功时`Ok`成员中的数据类型
-	`E`：失败时`Err`成员中返回的数据类型

###	直接处理

-	对`Result`值进行模式匹配，分别处理

	let f = File::open("hello.txt");
	let mut f = match f {
		Ok(file) => file,
		Err(error) => panic!("error:{:?}", error),
	}

-	使用`Result`上定义的方法（类似以上）

	-	`Result.unwrap()`
		-	`T = Ok<T>.unwrap()`
		-	`Err<E>.unwrap()`使用默认信息调用`panic`

	-	`Result.expect(&str)`
		-	`T = Ok<T>.expect(&str)`
		-	`Err<E>.expect(&str)`使用`&str`调用`!panic`

	-	`Result.unwrap_or_else`
	
			Result.unwrap_or_else(|err|{
				clojure...
			})

		-	`T = Ok<T>.unwrap_or_else()`
		-	`Err<E>.unwrap_or_else()`将`E`作为闭包参数调用
			闭包

	-	`Result.is_err()`
		-	`False = Ok<T>.is_err()`
		-	`True = Err<E>.is_err()`

###	传播错误（Propagating）

对`Result`对象进行匹配，提前返回`Err<E>`，需要注意返回值
类型问题，尤其是在可能存在多处潜在错误需要返回

	let f = File:open("hello.txt");
	let mut f match f {
		Ok(file) => file,
		Err(error) => return Err(error),
	}


`?`简略写法（效果同上）

	let mut f = File::open("hello.txt")?

-	`?`会把`Err(error)`传递给`from`函数（定义在标准库`From`
	trait中），将错误从转换为函数**返回值**中的类型，潜在
	错误类型都实现了`from`函数
-	`?`只能用于返回值为`Result`类型的函数内，因为其"返回值"
	就是`Err(E)`（如果有）

