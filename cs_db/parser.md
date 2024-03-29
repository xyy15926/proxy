---
title: Database Parser
categories:
  - Database
tags:
  - Database
  - Parser
date: 2019-04-09 14:46:49
updated: 2021-08-02 17:28:44
toc: true
mathjax: true
comments: true
description: Database Parser
---

##	*Parser*综述

*Parser*：分析器，将SQL语句切分为token，根据一定语义规则解析
成AST
#todo

###	查询计划/树

![sql_parser_ast](imgs/sql_parser_ast.png)

查询计划：由一系列内部操作符组成，操作符按照一定运算关系构成
查询的一个执行方案

-	形式上：二叉树
	-	树叶是每个单表对象
	-	两个树叶的父节点是连接操作符连接后的中间结果
	-	每个结点即临时“关系”

-	查询的基本操作：选择、投影、连接
	-	选择、投影的优化规则适用于*select-projection-join*
		操作和非SPY（SPY+Groupby）操作
	-	连接操作包括两表连接、多表连接

###	结点类型

-	单表结点：从物理存储到内存解析称逻辑字段的过程
	-	考虑数据获取方式
		-	直接IO获取
		-	索引获取
		-	通过索引定位数据位置后再经过IO获取相应数据块

-	两表结点：内存中元组进行连接的过程
	-	完成用户语义的局部逻辑操作，完成用户全部语义需要配合
		多表连接顺序的操作
	-	不同连接算法导致连接效率不同
	-	考虑两表
		-	连接方式
		-	代价
		-	连接路径

-	多表中间结点：多个表按照“最优”顺序连接过程
	-	考虑代价最小的“执行计划”的多表连接顺序

###	*Schema Catalog*

元数据信息：表的模式信息

-	表的基本定义：表名、列名、数据类型
-	表的数据格式：json、text、parquet、压缩格式
-	表的物理位置


