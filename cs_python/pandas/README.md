---
title: Pandas约定
tags:
  - Python
  - Pandas
categories:
  - Python
  - Pandas
date: 2019-03-29 18:39:51
updated: 2019-03-29 18:39:51
toc: true
mathjax: true
comments: true
description: Pandas约定
---

##	常用参数说明

-	函数书写声明同Python全局
-	以下常用参数如不特殊注明，按此解释

###	DataFrame

-	`axis=0/1/"index"/"columns"`
	-	含义：作用方向（轴）
	-	默认：`0/"index"`，一般表示row-wise（行变动）方向

-	`inplace=False/True`
	-	含义：是否直接在原对象更改
	-	默认：`False`，不更改，返回新DF对象（为`True`时无返回值）
	-	其他
		-	大部分df1.func()类型函数都有这个参数

-	`level=0/1/level_name...`
	-	含义：用索引层级
	-	默认：部分默认为`0`（顶层级）（也有默认为底层级），
		所以有时会如下给出默认值
		-	`t`（top）：顶层级`0`（仅表意）
		-	`b`（bottom）：底层级`-1`（仅表意）
		-	默认值为`None`表示所有层级

##	Pandas非必须依赖包

###	文件相关

-	Excel
	-	`xlrd/xlwt`：*xls*格式读写，速度较快
	-	`openpyxl`：*xlsx*格式读写，速度较慢

##	Pandas版本

-	`0.22.x`
	-	`flaot`类型可作为`Categorical Index`成员，不能被用于
		`loc`获取值

-	`l.1.1`
	-	`astype`方法不支持`pd.Timestamp`类型，只能用
		`"datetime64"`替代

-	`ALL`
	-	`Category Series`作为`groupby`聚集键时，类别元素都会
		出现在聚集结果中，即使其没有出现在seris值中
	-	`set`、`frozenset`被认为是*list-like*的indexer，所以
		在索引中的`frozenset`无法用一般方法获取

		```python
		df.iloc[list(df.index).index(fronzenset)]
		```


