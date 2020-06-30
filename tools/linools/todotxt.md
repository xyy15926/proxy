---
title: Todo.txt
tags:
  - 工具
  - Schedule
categories:
  - 工具
date: 2020-06-30 00:48:32
updated: 2019-06-30 00:48:32
toc: true
mathjax: true
comments: true
description: Todo.txt格式 - 纯文本todo list格式
---

##	Todo.txt格式

-	todo.txt格式使用纯文本存储、表示任务
-	存储任务的文本文件中每行表示一项task
-	todo.txt类清单管理应用因此一般具备如下特点
	-	待完成任务、已完成任务分为两个文件分别存储
	-	可以自由修改任务存储文件以修改任务内容

###	基本要素

![todotxt_format_description](imgs/todotxt_format_description.png)


-	任务结构：各部分之间使用空格分隔
	-	`x`：任务完成标识符
	-	优先级：`([A-Z])`
	-	完成日期、创建日期：`Y-m-d`格式
	-	任务描述：任务主体部分

-	任务描述部分可以包含各种类型的tag，tag格式上可以放在任务
	任何部分，但习惯上放在末尾
	-	*project*：`+`标识
	-	*context*：`@`标识
	-	*[key]:[value]*：metadata键值对，*key*表示键值对含义
		（中间一般不用空格分隔，便于处理）

> - 常用特殊metadata键值对
> > -	时间戳：`t:Y-m-d`
> > -	截至日期：`due:Y-m-d`

##	Todo.txt工具

###	todo.txt-cli

*todo.txt-cli*：基于shell脚本管理todo.txt文件

-	只提供基本todo.txt功能，可以通过自行添加脚本插件方式扩展
	-	插件文件即普通可执行脚本文件，`todo.sh`会将命令参数
		传递给相应脚本文件
	-	`todo.sh [action]`将直接调用相应`action`名称脚本
		（可建立同名文件夹、文件管理，同名文件夹文件被调用）

-	使用两个文件分别存储待完成任务、已完成任务
	-	待完成任务文件中可以自行`x`标记完成，然后使用命令
		归档至已完成任务文件中

> - <https://github.com/todotxt/todo.txt-cli>

###	topydo

*topydo*：基本python脚本管理todo.txt文件

-	提供了cli、prompt、column三种模式
-	原生支持以下标签
	-	due、start日期
	-	管理任务之间依赖管理
	-	重复任务


