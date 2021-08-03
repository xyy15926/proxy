---
title: STL String
categories:
  - C/C++
  - STL
tags:
  - C/C++
  - STL
date: 2019-03-21 17:27:37
updated: 2021-08-02 18:01:46
toc: true
mathjax: true
comments: true
description: String
---

字符串：理论上是指特定字符序列

-	声明`string`类型变量时，一般赋予字符串字面值作为初始值
-	字符串长短、索引类型默认是`size_t`类型，在`<string>`类库
	中已经定义

##	`<string>`

###	操作

-	`+`
-	`+=`
-	`==`
-	`!=`
-	`<`
-	`<=`
-	`>`
-	`>=`

###	读字符串内容

-	`.length()`
-	`.at(k)`：返回值可以用于赋值
-	`.substr(pos, n)`
-	`.compare(str)`
-	`.find(pattern, pos)`

###	修改接收方字符串内容

-	`.erase(pos, n)`
-	`.insert(pos, str)`
-	`.replace(pos, n, str)`

###	C风格

-	`string(carray)`
-	`string(n, ch)`
-	`.c_str()`

