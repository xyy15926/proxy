---
title: NDArray子类
categories:
  - Python
  - Numpy
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-02-18 17:10:18
updated: 2021-02-18 17:10:18
toc: true
mathjax: true
description: NumPy模块中`ndarray`子类介绍，基本包括
  掩码类、记录类等
---

##	子类相关钩子属性、方法

###	`__array__`方法

-	`class.__array_ufunc__(ufunc, method, *inputs, **kwargs)`
	-	功能：供自定义以覆盖numpy中ufunc行为
		-	返回操作结果，或`NotImplemented`
			（将此方法置`None`）
	-	参数
		-	`ufunc`：被调用的ufunc对象
		-	`method`：字符串，指示调用的`ufunc`对象的方法
		-	`inputs`：`ufunc`顺序参数元组
		-	`kwargs`：`ufunc`关键字参数字典

	> - *Ufunc*、与`__array_ufunc__`关系参见ufunc部分

-	`class.__array_function__(func,types,args,kwargs)`
	-	参数
		-	`func`：任意callable，以`func(*args, **kwargs)`
			形式调用
		-	`types`：来自实现`
		-	`args`、`kwargs`：原始调用的参数

-	`class.__array__finalize(obj)`
	-	功能：构造之后更改`self`的属性
		-	在为`obj`类型数组分配空间时调用
	-	参数
		-	`obj`：`ndarray`子类

-	`class.__array_prepare__(array,context=None)`
	-	功能：在ufunc计算前，将ouput数组转换为子类实例、
		更新元数据
		-	调用任何ufunc前，在最高优先级的input数组，或指定
			的output数组上调用，返回结果传递给ufunc
		-	默认实现：保持原样

-	`class.__array_wrap__(array,context=None)`
	-	功能：在将结果返回给用户前，将output数组转换为子类
		实例、更新元信息
		-	ufunc计算结果返回给用户前，在最高优先级的output
			数组、或指定output对象上调用
		-	默认实现：将数组转换为新

-	`class.__array__([dtype])`
	-	功能：若output对象有该方法，ufunc结果将被写入其
		返回值中

> - 若ufunc中所有`__array_ufunc__`返回`NotImplemented`，那么
	`raise TypeError`

###	`__array__`属性

-	`class.__array_priority__`
	-	功能：决定返回对象的数据类型（有多种可能性时）
		-	默认值：`0.0`

##	Matrix

###	`np.matrix`

Matrix对象：继承自`ndarray`，具有`ndarray`的属性、方法

-	Matrix对象的特殊行为
	-	维数始终为2
		-	`.ravel()`仍然二维
		-	*item selection*返回二维对象
	-	数学操作
		-	覆盖乘法为矩阵乘法
		-	覆盖幂次为矩阵幂次
	-	属性
		-	默认`__array_priority__`为`10.0`

> - Matrix类被设计用于与`scipy.sparse`交互，建议不使用
> - `np.mat`是`np.matrix`别名

####	Matrix对象property属性

|Property|Desc|
|-----|-----|
|`matrix.T`|转置|
|`matrix.H`|复数共轭|
|`matrix.I`|逆矩阵|
|`matrix.A`|返回`ndarray`|

####	Matrix创建

|Routine|Desc|
|-----|-----|
|`np.mat(data[,dtype])`|创建矩阵|
|`np.matrix(data[,dtype,copy])`|不建议使用|
|`np.asmatrix(data[,dtype])`|将数据转换为矩阵|
|`np.bmat(obj[,ldict,gdict])`|从字符串、嵌套序列、数组中构建|

-	`mp.bmat`：可使用Matlab样式字符串表示法创建Matrix
	-	空格分割列
	-	`;`分割行

###	`np.matlib`

-	`numpy.matlib`模块中包含`numpy`命名空间下所有函数
	-	返回`matrix`而不是`ndarray`
	-	`matrix`被限制为小于2维，会改变形状的函数可能无法
		得到预期结果

> - `np.matlib`是为了方便矩阵运算的模块

##	`np.char`

###	`np.chararray`

-	`np.chararray`类：`string_`、`unicode_`数据类型的增强型
	数组，继承自`ndarray`
	-	继承由*Numarray*引入的特性：项检索和比较操作中，数组
		元素末尾空格被忽略
	-	定义有基于元素的`+`、`*`、`%`的操作
	-	具有所有标准`string`、`unicode`方法，可以逐元素执行

|Routine|Function Version|
|-----|-----|
|`char.array(obj[,itemsize,...])`||
|`char.asarray(obj[,itemsize,...])`|转换输入为`chararray`，必要时复制数据|
|`chararray(shape[,itemsize,unicode,...])`|不应直接使用此构造函数|

> - `np.chararray`类是为了后向兼容*Numarray*，建议使用
	`object_`、`string_`、`unicode_`类型的数组替代，并利用
	`numpy.char`模块的自由函数用于字符串快速向量化操作

###	NDArray Char Routine

-	`np.char`/`np.core.defchararray`模块为`np.string_`、
	`np.unicode_`类型的数组提供向量化的字符串操作
	-	基于标准库中`string`、`unicode`的方法

####	字符串操作

|Routine|Function Version|
|-----|-----|
|`char.add(x1,x2)`||
|`char.multiply(a,i)`||
|`char.mod(a,values)`|`%`格式化（`str.__mod__`为`%`调用方法）|
|`char.capialize(a)`|首字符大写|
|`char.title(a)`|单词首字符大写|
|`char.center(a,width[,fillchar])`|`a`居中、`fillchar`填充字符串|
|`char.ljust(a,width(,fillchar))`|`a`靠左|
|`char.rjust(a,width(,fillchar))`|`a`靠左|
|`char.zfill(a,width)`|`0`填充左侧|
|`char.char.decode(a[,encoding,errors])`||
|`char.char.encode(a[,encoding,errors])`||
|`char.char.expandtabs(a[,tabsize])`|替换tab为空格|
|`char.join(sep, seq)`||
|`char.lower(a)`||
|`char.upper(a)`||
|`char.swapcase(a)`||
|`char.strip(a[,chars])`||
|`char.lstrip(a[,chars])`||
|`char.rstrip(a[,chars])`||
|`char.partition(a,sep)`|从左至右切分一次，返回三元组|
|`char.rpartition(a,sep)`|从右至左切分一次|
|`char.split(a[,sep,maxsplit])`|从左至右切分`maxsplit`次，返回列表|
|`char.rsplit(a[,sep,maxsplit])`||
|`char.splitlines(a[,keepends])`|切分行，即`\n`为切分点|
|`char.replace(a,old,new[,count])`||

####	Camparison

|Function|Desc|
|-----|-----|
|`equal(x1,x2)`||
|`greater(x1,x2)`||
|`less(x1,x2)`||
|`not_equal(x1,x2)`||
|`greater_equal(x1,x2)`||
|`less_equal(x1,x2)`||
|`compare_chararrays(a,b,com_op,rstrip)`|`com_op`指定比较方法|

####	字符串信息

|Function|Desc|
|-----|-----|
|`count(a,sub[,start,end])`|统计不重叠`sub`出现次数|
|`startwith(a,prefix[,start,end])`||
|`endswith(a,suffix[,start,end])`||
|`find(a,sub[,start,end])`|返回首个`sub`位置，不存在返回`-1`|
|`rfind(a,sub[,start,end])`|从右至左`find`|
|`index(a,sub[,start,end])`|同`find`，不存在`ValueError`|
|`rindex(a,sub[,start,end])`|从右至左`index`|
|`isalpha(a)`||
|`iaalnum(a)`||
|`isdecimal(a)`||
|`isdigit(a)`||
|`islower(a)`||
|`isnumeric(a)`||
|`isspace(a)`||
|`istitle(a)`|是否各单词首字符大写|
|`isupper(a)`||
|`str_len(a)`||

##	`np.rec`

-	`np.rec`/`np.core.records`

###	`np.recarray`

-	`np.recarray`类：允许将结构化数组的字段作为属性访问

|Routine|Function Version|
|-----|-----|
|`np.recarray`|创建允许属性访问字段的`ndarray`|
|`np.record`|允许使用属性查找字段的数据类型标量|

###	Record Arrays

|Routine|Function Version|
|-----|-----|
|`core.records.array(obj[,dtype,shape,...])`|从多类型对象中创建|
|`core.records.fromarrays(arrayList[,dtype,...])`|从数组列表中创建|
|`core.records.fromrecords(recList[,dtype])`|从文本格式的records列表创建|
|`core.records.fromstring(datastring[,dtype,...])`|从二进制数据字符串中创建只读|
|`core.records.fromfile(fd[,dtype,shape,...])`|从二进制文件中创建|


##	`np.ma`

###	`ma.MaskedArray`

-	`ma.MaskedArray`：掩码数组，是`np.ma`核心，`ndarray`子类
	-	`ma.MaskedArray`由标准`np.ndarray`和掩码组成

-	掩码数组`.mask`
	-	掩码可以被设置为*hardmask*、*softmask*，由只读属性
		`hardmask`指定
		-	*hardmask*：无法修改被遮蔽值
		-	*softmask*：可修改被遮蔽值，并恢复被遮蔽状态
	-	`.mask`可以被设置
		-	为bool数组，指示各位置元素是否被遮蔽
		-	`ma.maskded/ma.unmask/True/False`，设置掩码数组
			整体是被遮蔽

> - `ma.nomask`是`np.bool_`类型的`False`，`ma.masked`是特殊
	常数
> - `ma.MaskType`是`np.bool_`别名

> - <https://www.numpy.org.cn/reference/arrays/maskedarray.html>
> - <https://www.numpy.org.cn/reference/routines/ma.html>

####	属性

|Attr|Desc|
|-----|-----|
|`.hardmask`|硬掩码标志|
|`.data`|值数组|
|`.mask`|掩码数组、`ma.unmask`、`ma.masked`|
|`.recordmask`|项目中命名字段全遮蔽则遮蔽|

####	创建掩码数组

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.MaskedArray(data[,mask,dtype,...])`|类|无|
|`ma.masked_array(data[,mask,dtype,...])`|`MaskedArray`别名|无|
|`ma.array(data[,dtype,copy,...])`|构造函数|无|
|`ma.frombuffer(buffer[,dtype,count,offset])`||无|
|`ma.fromfunction(function,shape,dtype)`||无|
|`ma.fromflex(fxarray)`|从有`_data`、`_mask`字段的结构化`fxarray`中创建|无|
|`copy(a[,order])`|||

#####	Ones and Zeros

|Routine|Function Version|
|-----|-----|
|`ma.empty(shape[,dtype,order])`|无初始化|
|`ma.empty_like(prototype[,dtype,order,subok,...])`|shape、类型同`prototype`|
|`ma.ones(shape[,dtype,order])`||
|`ma.zeros(shape[,dtype,order])`||
|`ma.masked_all(shape[,dtype])`|所有元素被屏蔽|
|`ma.masked_all_like(shape[,dtype])`||

###	MaskedArray Routine

-	`np.ma`模块下的函数、`ma.MaskedArray`方法和`ndarray`
	类似，但行为可能不同
	-	`np`命名空间下部分函数（`hstack`等）应用在
		`MaskedArray`上
		-	操作时忽略`mask`（即会操作被遮罩元素）
		-	返回结果中`mask`被置为`False`

> - 这里仅记录`ma`模块中额外、或需额外说明部分

####	数组检查

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.all(a[,axis,out,keepdims])`|全遮蔽时返回`ma.masked`||
|`ma.any(a[,axis,out,keepdims])`|存在遮蔽时返回`ma.masked`||
|`ma.count(arr,[axis,keepdims])`|沿给定轴统计未被遮罩元素数量||
|`ma.count_masked(arr,[axis])`|沿给定轴统计被遮罩元素数量||
|`ma.nonzero(a)`|非0、未屏蔽元素索引||
|`ma.is_mask(m)`|是否为标准掩码数组|无|
|`ma.is_masked(x)`|是否包含遮蔽元素||

####	获取、创建、修改掩码

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.getmask(a)`|返回掩码、或`ma.nomask`、`ma.masked`|`.mask`属性|
|`ma.getmaskarray(arr)`|返回掩码、或完整`False`数组|无|
|`ma.make_mask(m[,copy,shrink,dtype])`|从数组创建掩码|无|
|`ma.make_mask_none(newshape[,dtype])`|创建给定形状掩码|无|
|`ma.make_mask_descr(ndtype)`|为给定类型的创建掩码类型|无|
|`ma.mask_rowcols(a[,axis])`|遮蔽包含遮蔽元素的`axis`方向分量|无|
|`ma.mask_rows(a[,axis])`|缺省为`0`的`mask_rowcols()`|无|
|`ma.mask_cols(a[,axis])`|缺省为`1`的`mask_rowcols()`|无|
|`ma.mask_mask_or(m1,m2[,copy,shrink])`|掩码或|无|
|`ma.harden_mask(a)`|||
|`ma.soften_mask(a)`|||
|`.shrink_mask()`|无|尽可能缩小掩码|
|`.share_mask()`|无|复制掩码，并设置`sharedmask=False`|

####	获取、创建索引

-	索引非结构化掩码数组
	-	mask为`False`：返回数组标量
	-	mask为`True`：返回`ma.masked`

-	索引结构化掩码数组
	-	所有字段mask均为`False`：返回`np.void`对象
	-	存在字段mask为`True`：返回零维掩码数组

-	切片
	-	`.data`属性：原始数据视图
	-	`.mask`属性：`ma.nomask`或者原始mask视图

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.nonzero(a)`|未屏蔽、非0元素索引||
|`ma.mr_[]`|沿第1轴concate切片、数组、标量，类`np.r_[]`|无|
|`ma.flatnotmasked_contiguous(a)`|展平后未遮蔽切片|无|
|`ma.flatnotmasked_edges(a)`|展平后首个、末个未遮蔽位置|无|
|`ma.notmasked_contiguous(a[,axis])`|沿给定轴，未遮蔽切片|无|
|`ma.notmasked_edges(a[,axis])`|沿给定轴，首个、末个未遮蔽位置|无|
|`ma.clump_masked(a)`|展平后遮蔽切片|无|
|`ma.clump_unmasked(a)`|展位后未遮蔽切片|无|

-	`ma.mr_[]`类似`np.r_[]`，但`np.r_[]`返回结果掩码被置为
	`False`，而`ma.mr_[]`同时也操作掩码

####	获取、修改值

-	仅访问有效数据
	-	对掩码mask取反作为索引`~X.mask`
	-	使用`.compressed`方法得到一维`ndarray`

	```python
	print(X[~X.mask])
	print(X.compressed())
	```

-	访问数据
	-	通过`.data`属性：可能是`ndarray`或其子类的视图
		-	等同于直接在掩码数组上创建`ndarray`或其子类视图
	-	`__array__`方法：`ndarray`
	-	使用`ma.getdata`函数

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.getdata(a[,subok])`|返回掩码数组数据|`.data`属性|
|`ma.fix_valid(a[,mask,copy,fill_value])`|替换`a`中无效值，并遮盖|无|
|`ma.masked_equal(x,value[,copy])`||无|
|`ma.masked_greater(x,value[,copy])`||无|
|`ma.masked_greater_equal(x,value[,copy])`||无|
|`ma.masked_inside(x,v1,v2[,copy])`||无|
|`ma.masked_outside(x,v1,v2[,copy])`||无|
|`ma.masked_invalid(x[,copy])`||无|
|`ma.masked_less(x,value[,copy])`||无|
|`ma.masked_less_equal(x,value[,copy])`||无|
|`ma.masked_not_equal(x,value[,copy])`||无|
|`ma.masked_values(x,value[,rtol,atol,...])`||无|
|`ma.masked_object(x,value[,copy,shrink])`|类`masked_values`，适合值类型为`object`时|无|
|`ma.masked_where(condition,a[,copy])`|按`condition`遮蔽指定值|无|

####	其他属性、方法

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.common_fill_value(a,b)`|若`a,b`填充值相同则返回，否则返回`None`|无|
|`ma.default_fill_value(obj)`|默认填充值|无|
|`ma.maximum_fill_value(obj)`|对象类型决定的最大值|无|
|`ma.minimum_fill_value(obj)`||无|
|`ma.sef_fill_value(a,fill_value)`|||
|`.get_fill_value()`/`.fill_value`|无||
|`ma.allequal(a,b[,fill_value])`|若`a,b`元素均相等，则使用`fill_value`填充||

###	`np.ma`运算

-	掩码数组支持代数、比较运算
	-	被遮蔽元素不参与运算，元素在运算前后保持不变
	-	掩码数组支持标准的*ufunc*，返回掩码数组
		-	运算中任意元素被遮蔽，则结果中相应元素被遮蔽
		-	若*ufunc*返回可选的上下文输出，则上下文会被处理，
			且无定义结果被遮蔽

-	`np.ma`模块中对大部分ufunc有特别实现
	-	对于定义域有限制的一元、二元运算，无定义的结果会
		自动mask

	```python
	ma.log([-1, 0, 1, 2])
	```

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`ma.anom(a[,axis,dtype])`|沿给定轴计算与算数均值的偏差||

##	`np.memmap`

-	`np.memmap`：内存映射文件数组，使用内存映射文件作为数组
	数据缓冲区
	-	对大文件，使用内存映射可以节省大量资源

-	方法

	|Method|Desc|
	|-----|-----|
	|`np.memmap(filename[,dtype,mode,shape])`|创建存储在磁盘文件的内存映射数组|
	|`np.flush()`|flush内存数据至磁盘|

##	标准容器类

-	`np.lib.user_array.container`
	-	为向后兼容、作为标准容器类而引入
	-	其中`self.array`属性是`ndarray`
	-	比`ndarray`本身更方便多继承

-	类、方法、函数

	|Method|Desc|
	|-----|-----|
	|`np.lib.user_array.container(data[,...])`|简化多继承的标准容器类|




