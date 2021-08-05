---
title: Numpy Readme
categories:
  - Python
  - Numpy
tags:
  - Python
  - Numpy
  - Readme
date: 2021-01-31 15:49:04
updated: 2021-08-04 16:51:12
toc: true
mathjax: true
description: 
---

##	常用参数说明

-	函数书写说明同Python全局
-	以下常用参数如不特殊注明，按照此解释

###	NDArray常用参数

####	基本数组参数

-	`size=None(1)/int/Tuple[int]`

-	`shape=None/int/Tuple[int]`
	-	含义：NDArray形状
		-	`int`：1维时可直接用整数表示shape
		-	`tuple`：高维至低维指定哥维度大小
			-	`-1`：由整个size、其余维度推断该维度大小
	-	默认：`None`、`1`

-	`dtype=None/str/list/dict/np.dtype/...`
	-	含义：指定输出数组数据类型
		-	`None`：保证精度情况下自动选择数据类型
		-	`str`、`list`、`dict`：可转换为数据类型
		-	`np.dtype`：`np.dtype`实例
	-	默认值：`None`，有内部操作，选择合适、不影响精度类型

-	`order="K"/"C"/"F"/"A"`
	-	含义：指定数组对象（输出）内存布局、迭代顺序
		-	`"C"`：C-contiguous风格，行优先
		-	`"F"`：Fortran-contiguous风格，列优先
		-	`"A"`：除非所有参数数组均为Fortran风格，否则
			为C风格
		-	`"K"`：尽量贴近已有内存布局，原为"C"/"F"方式则
			保持不变，否则选择较接近的风格
	-	默认值："C"/"K"

-	`casting="same_kind","no","equiv","safe","unsafe"`
	-	含义：类型转换规则
		-	`no`：不允许任何类型转换
		-	`equiv`：仅允许字节顺序改变
		-	`safe`：仅允许可保证数据精度的类型转换
		-	`same_kind`：只能允许`safe`或同类别类型转换
		-	`unsafe`：允许所有类型转换
	-	numpy 1.10及以上版本，缺省为`"same_kind"`

####	结果参数

-	`out=None/Tuple[Array]/Array`
	-	含义：保存结果的变量
		-	`None`：由函数自行分配空间
		-	`tuple`：需要存储多个输出结果的变量元组
		-	`Array`：仅需要保存单个输出结果的变量元组
	-	默认：`None`
	-	函数自行分配空间不会初始化，即若其中某些元素未被设置
		值，则其值不可预测，如
		-	`where`非`True`时，`False`对应元素

-	`keepdims=False/True`
	-	含义：是否维持原维数
		-	`True`：保留本应被缩减的维度，并设置维度长为1
			-	保证结果和输入操作数广播兼容
		-	`False`：不保持维数不变
	-	默认：`False`

-	`subok=True/False`
	-	含义：是否允许数组子类作为输出
		-	`True`：允许
		-	`False`：不允许
	-	默认：`True`


####	标记参数

-	`where=True/False/Array[bool]`
	-	含义：指示符合条件、需执行操作的bool map
		-	`True`：广播为全`True`，所有元素
		-	`False`：广播为全`False`，所有元素都不
		-	`Array[bool]`：`True`表示对应位置的元素满足条件
			（需要和输入操作数广播兼容）

-	`weekmask="1111100"/str/list`
	-	含义：指示一周内工作日
		-	字符串
			-	`1`、`0`按顺序表示周一到周日为、非工作日
			-	空白符、驼峰分割周一至周日全称或缩写
		-	列表：`0`、`1`按顺序表示周一到周日为、非工作日

-	`condition=Array[bool,int]`
	-	含义：指示符合条件、需要执行操作的bool map
		-	`Array[bool]`：`True`表示对应位置的元素满足条件
		-	`Array[int]`：根据是否为`0`转换为bool数组



