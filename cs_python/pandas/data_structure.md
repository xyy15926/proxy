---
title: Pandas数据结构
tags:
  - Python
  - Pandas
categories:
  - Python
  - Pandas
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Pandas数据结构
---

##	DataFrame

可以看作包含两个Index类（index、columns）、一个二维ndarray类
（values）

-	二维values + 结构化index + 结构化columns
	-	values自己还有两根只能使用integer indice的轴
-	结构同数据库表相似：列为属性，行为个体

```python
DF = pd.DataFrame(
	data=ndarray/{col_name: val}/DF/array-like,
	index=Index/array-like,
	columns=Index/array-like,
	dtype=None/dtype,
	copy=False)

	# `data=dict`：可以通过`columns`仅使用部分键值对创建
	# `copy`：仅影响`data`为DF/ndarray时，默认不使用拷贝
```
###	DF行列逻辑

可以通过`DF = df1.T`作转制更改行列逻辑

####	获取数据

#####	列优先

```python
Ser = df1[col_name]
Ser = df1.col_name
	# 取列
Ser = df1[col_level_0, col_level_1,...]
Ser = df1[(col_level_0, col_level_1,...)]
Ser = df1.col_level_0.col_level_1
	# 对层级索引取列

Val = df1[col_name][index_name]
Val = df1.col_name.index_name
	# 取具体值
	# 属性方式，要求`*_name`为字符串
	# `.`、`[]`应该是覆盖过

Val = df1[col_level_0, col_level_1,...]\
	[index_level_0, index_level_1,...]

Val = df1[(col_level_0, col_level_1,...)]\
	[index_level_0, index_level_1,...]

Val = df1.col_level_0.col_level_1....\
	.index_level_0.index_level_1...
	# 对层级索引取值

DF = df1[cond1 & cond2 &...]
DF = df1[cond1][cond2]...
	# `condX`为Ser(bool)
	# 两种讲道理应该没差
```

#####	行优先

######	`.loc[]`

行优先,逻辑和df1[]列优先类似

```python
Ser = df.loc[index_name]
	# 取行
Ser = df.loc[index_level_0, index_level_1,...]
Ser = df.loc[(index_level_0, index_level_1,...)]
	# 层级索引取行Ser/DF

Val = df1.loc[index_name, column_name]
	# 还可以这样的取值
	# 不建议使用，容易照成混淆
 # Val = df1.loc.index_name`
	# 不可

Val = df1.loc[index_level_0, index_level_1,...]\
	[col_level_0, col_level_1,...]
	# 层级索引时，index、col不能混在一起

Val = df1.loc[(index_level_0, index_level_1,...)]\
	[(col_level_0, col_level_1,...)]
```

######	`.iloc[]`

indices locate，应视为对value（ndarray）进行操作，index和
columns的结构对其没有任何影响

-	在层级索引情况下，仍然返回Ser

```python
Ser = df1.iloc[indice]
	# 返回values第indice行Ser对象
Val = df1.iloc[index_indice, col_indice]
	# 取得values第index_indice行、第col_indice列元素
```

######	`.ix[]`

`.loc`、`.iloc`的封装，不建议使用

-	优先使用`.loc`，除非参数为int、且Index不是int类型

```python
Ser = df1.ix[index]
```

#####	行优先快速版本

**只能**、**必须**取一个值
-	不能用于获取切片
-	对于多重索引必须将Index、Columns所有level全部指定

-	`.iat`
	```python
	Val = df1.iat[index_indice, col_indice]
	```

-	`.at`
	```python
	Val = df1.at[index_name, col_name]
		# 单索引
	Val = df1.at[(index_level_0, index_level_1,...), (col_level_0, col_level_1,...)]
		# 多重索引必须指定全部level保证只取到一个值
	```

###	切片

####	Values切片

切片对象是values

```python
DF = df1.iloc[irow_start: irow_end, icol_start: icol_end]
	# 对values的切片，参数都是indices
	# 这个应该就是ndarray切片操作，不包括上限
	# 如果只对行切片，可以省略`.iloc`，但不建议，因为这个
		# 同时也可以表示Index切片（优先）
```

####	Index切片

切片对象是index，包括上限

#####	全切片

```python
DF = df1.loc[
	(index_0, index_1,...): (index_0, index_1,...),
	(col_0, col_1,...): (col_0, col_1,...)]
	# `.loc`可以替换为`.ix`，但不能删除，不建议使用
```

#####	行切片

```python
DF = df1.loc[[(index_0, index_1,...),...]]
DF = df1.loc[(index_0, index_1,...): (index_0, index_1,...)]
	# index_level可以不用指定到最低level，
	# 同样的，`.loc`可以替换为`.ix`，但不建议使用
```

#####	列切片

```python
DF = df1[[col_name,...]]
DF = df1.loc[:, (col_0,...): (col_0,...)]
	# `.loc`可以替换为`.ix`，但是不能删除，不建议使用
	# 列切片没有`:`语法，只能通过设置行切片为`:`得到
```

###	DF数据共享逻辑

DF数据（values）共享的逻辑

-	一般尽量共享数据，直至无法处理（数据同时增加/删除行、列）
-	有些方法会有`copy`参数，可以显式控制是否拷贝副本
	-	如`.reindex`默认拷贝副本

```python
df1_T = df1.T
	# 两个此时共享数据，对任一的更改会反映在另一者
df1_T["new_col_1"] = [ ]
	# 添加新列后，两者仍然共享公共部分数据，只是`df1`中无法
		# 访问新列
df1["new_col_2"] = [ ]
	# 此时两者数据均独立

	# 类似的`del`删除列也是如此逻辑
```

##	Index 索引

使用integer作为index时注意df1.ix[]的逻辑

###	MultiIndex 层级索引

层级索引允许以低维度形式表示高纬度数据

-	层级索引可以使用tuple形式表示：`(level_0, level_1,...)`
	-	需要注意区分和tuple本身作为index
		-	打印时可以tuple有括号，而层级索引没有
		-	层级索引有时可以省略括号

####	`from_arrays`

```python
Index = pd.MultiIndex.from_arrays(
	arrays([[],[]]),
	sortorder=None/int,
	names=None/[])

Index = [
	level_0_list,
	level_1_list,...]
	# 隐式构建层级索引，各list长度相同，其按顺序组合
	# 可以在：DF构造参数、给DF对象Index赋值等使用
	# 应该是可以看作pandas使用`from_arrays`处理
```

-	说明：将arrays转换为MultiIndex

-	参数
	-	`arrays`：包含多个list作为各个level索引
		-	各list按照传递顺序决定level
		-	**不会自动合并**不连续labels（否则需要交换数据位置）
	-	`sortorder`：sortedness级别？
	-	`names`：level名

####	`from_tuples`

```python
Index = pd.MultiIndex.from_tuples(
	tuples=[tuple-like],
	sortorder=None/int,
	names=None)
```

-	说明：将`tuples`转换为MultiIndex

-	参数
	-	`tuples`：每个tuple为一个index，按照tuple中元素顺序
		决定各元素level

####	`from_product`

```python
Index = pd.MultiIndex.from_product(
	iterables([[]]/[iterables]),
	sortorder=None/int,
	names)
```

-	说明：对`iterables`元素作product（积）作为MultiIndex



2.	Series：可以看作是包含一个Index类（index，存放标签）、一个一维ndarray类（values，存放数据）

	a.	ser=pd.Series(data=np.darray/dict, index=list)

	b.	Series对象可以处理标签不一致的数据，但是只有标签的交集才能得到有意义的结果，其余为NaN

	c.	其余性质类似于DataFrame对象

3.	Index属性#todo

	a.	index属性

		1.	df1.columns=[]：更改列名称，index同

		2.	df1.columns.names=[]：更改列名，index同

4.	pandas库中的其他一些问题

	a.	数据类型转换：Series对象和DF对象在运算过程中dtype类型可能发生"无意义"的转换

		1.	dtype=i8的对象之间的+、-结果为dtype=f8类型的对象（当然这个可能是保持和\的一致性）

		2.	SeriesObj.reindex(new_index)会"可能"会改变原有数据类型（由i8->f8）（有增加新index时）

