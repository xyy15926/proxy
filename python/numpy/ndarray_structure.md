#	Numpy数据结构

##	ndarray

numpy的核心ndarray对象多维数组

-	同质元素（dtype）组成
-	大小固定
-	多维数组的概念类似于C中的概念，是规则的立方体

```python
NDA = np.array(
	object(array-list),
	dtype = None/dtype/[(fieldname, dtype)]/
		{"names" : [fieldnames], "fields": [dtype]}
	copy = True,
	order = "C"/"A"/"K"/"F",
	subok = False,
	ndmin = int
)
	# `dtype`
		# None：按照需要的minimium type设置
		# dtype：设为为dtype类型
		# [(fieldname, dtype)]：设置结构化数组fieldname、dtype
		# {"names": [fieldnames], "fields": [dtype]}：同上
	# `copy`：默认拷贝`object`
		# 如果设置为False，在`__array__`返回拷贝、`object`
			# 为nested seq、需要满足其他参数时仍copy
	# `order`：指定内存中NDA对象存储方式，默认"K"方式
		# K：保持F/C存储方式不变，否则选择F/C中相近者
		# C：列优先存储（强制，`copy=False`无效）
		# F：列优先存储（Fortran存储方式）
			# （强制，`copy=False`无效）
		# A：除非为F方式，否则C方式
	# `subok`：默认NDA被强制转换为base-class，否则sub-class
		# 会被传递
	# `ndmin`：NDA对象最小维度
```

###	副本、视图

####	视图

NDA对象通过`=`直接赋值一般都是视图 

-	多个视图共享同一个数据区，对任一个视图的数据更改会影响
	到其他对象
-	但是多个视图有一些独立的属性（如维数），类似于`C`中包含
	`data_pointer`的复制
-	NDA对象的切片也是视图（python list类型的切片不是）

####	副本

可以通过`NDA.copy()`获得副本拷贝

###	bool数组

NDA对象可以直接进行bool运算得到bool元素的NDA

```python
NDA(bool) = nd1 < 0
	# 这个可以看作是将右值广播后进行比较

NDA[NDA(bool)]
	# NDA(bool)可以用来筛选元素
```
###	结构化数组

ndarray元素为特定结构tuple，类似于二维数据表

```python
NDA(tuple) = np.array(
	[tuple1, tuple2,...],
	dtype = ("dtype1, dtype2,...)/
		[(filedname1, dtype1), (filename2, dtype2),...]
)
	# `dtype`必须显式给出各个field数据类型，否则只构建普通
		# 2DA对象（tuple长度相同）或tuple为元素的1DA，而各个
		# field数据类型由np推断
	# `dtype`给出一个dtype也是表示普通NDA初始化，需要满足
		# tuple长度相等

		# `(str)`：各field名称为`f0`，`f1`
```

-	以看作是有一维的index为`str`的二维数组，可以通过
	`NDA[fieldname]`得到该field的一个ndarray

####	数据推断

numpy数据类型都是按最小风险推断

```python
string(U)< complex(c16)< float(f8)< int(i8)
```

-	str被推断为`< U_MAXLEN(unicode)`而不是`< S_MAXLEN(bytes)`
-	complex、float、int类型被推断为最精确类型

###	广播、向量化

实现对形状不完全相同的ndarrayObj的运算

####	条件

数组各维度兼容，即数组对应维度：

-	各维等长
-	长度为1：用已有值填充至该维度长度一致
-	长度为0（即该维度缺失）：这个其实应该看作是长度为1的特例

####	维度缺失

维度缺失应该视为**高维度**缺失，**低维度**无法缺失的

-	因为数组的维度排序是确定的，若需要跳过第二维（维度缺失）
	，需要将第二维长度设为1，否则所有维度level都会下降

-	所以应该将缺失维度视为维度长度为1，只是因为缺失是高维度
	缺失，可以省略

-	当然这样的考虑下数组维度无限，只需要考虑到长度长度“不为
	1”的最高维度

####	广播方式


具体来说就是将多个兼容的NDA“面”综合考虑后，用已有值扩展为
合适NDA“体”后运算。需要注意的是，数组维度level是
**左高->右低**，维度应该是**右对齐**，在**左边补齐**缺失维度
（长度为1）。

-	数组维度右对齐
-	检查兼容性
-	用1补齐缺失高维度（缺失m维度在最外层加上m层`[]`补齐）
-	确定最终NDA“体”各维度长度
-	**从底（右）到高（左）**依次递归复制该维度数据（即包含
	低于其维度`[]`内的所有数据）

```python

shape(3, 2, 2, 1) + shape(1, 3)
	-> shape(3, 2, 2, 1) + shape(1, 1, 1, 3)
	-> shape(3, 2, 2, 3) + shape(1, 1, 2, 3)
	-> shape(3, 2, 2, 3) + shape(1, 2, 2, 3)
	-> shape(3, 2, 2, 3) + shape(3, 2, 2, 3)
```

####	向量化

代码中可以省略部分显式循环，由numpy内部实现

-	将NDA对象视为向量，可以"直接"进行向量运算

###	函数


####	`stack`

```python
NDA = np.stack(
	arrays([NDA]),
	axis = 0/int,
	out = None/NDA
)

```
-	说明：沿着`axis`轴方向join `arrays`
	-	不是concatenate，这是是单纯的把相应维度用`[]`括起来，
		既是结构上的用`[]括起，也是NDA对象打印的扩起，从某些
		角度看，原`arrays`中元素未改变
	-	因此这个函数要求`arrays`对象shape完全相同，而不是
		只要求`join`轴方向“shape”相同
	-	一定会**增加1维度**，并且除了`axis = -1`不需要直觉上
		更改维度构图（在最高维join，增加维度为最低维），其他
		全需要
	-	可以通过`NDA[:, index, :, ..., :]`得到原`arrays`中
		元素，
		-	index为元素在`arrays`中index
		-	位于第`axis`位置

-	参数
	-	`axis`：最大取值为`arrays`维度，即`-1`效果
	-	`out`：存放结果的NDA，shape必须相符

####	`concatenate`

```python
NDA = np.concatenate(
	([array-like]),
	axis = 0/int,
	out = None
)
```

-	说明：在沿着`axis`轴方向concatenate `[array-like]`
	-	在相应维度`+`起来，即沿`axis`轴拆出然后合并
	-	因此只要求`concatenate`轴方向shape相同
	-	一定不会增加维度，是在维度上进行合并
	-	可以通过`NDA[:, start: end, :, ..., :]`得到原NDA
		-	`start:end`为相应NDA对象切片
		-	位于第`axis`位置

####	`x_stack`

这个系列的函数逻辑有点特殊

-	`tup`维度大于**函数**要求维度时，相当于`axis`取相应值的
	`concatenate`
-	`tup`维度小于**函数**要求维度时，相当于`axis`取相应值的
	`stack`

```python
NDA = np.vstack/np.row_stack(
	tup([NDA])
)
	# `axis = 1`

NDA = np.hstack/np.column_stack(
	tup([NDA])
)
	# `axis = 2`

NDA = np.dstack(
	tup([NDA])
)
	# `axis = 3`
	# 这个可以做到`stack`无法做到的事情
		# `stack`的`axis`参数至多只能取到`arrays`元素维度
			# `+1`（即取`-1`）
		# 此函数可以对1DA相当于可以取到`axis = 3`
```

####	`split`

```python
[NDA] = np.split(
	ary(NDA),
	indices_or_sections(int/array-like),
	axis = 0/int
)

[NDA] = np.array_split(
	ary(NDA),
	indices_or_sections(int/array-like),
	axis = 0/int
)
```

-	说明：沿`axis`轴split为[NDA]
	-	返回的每个NDA维度仍和原NDA相同

-	参数
	-	`indices_or_sections`
		-	int；均等分为`indices_or_sections`份
			-	若无法均分`split`会`raise ValueError`
			-	`array_split`仍然正常返回，最后几个NDA有缺
		-	array-like：以其元素为分界点split

####	`block`

```python
NDA = np.block(
	arrays(nested-list(NDA))
)
```

-	说明：“矩阵”（可能为高维）分块的逆运算
	-	按照`arrays`给出的结构将**block**合并
	-	其中NDA不会进行广播，要求`arrays`中元素shape符合要求

```python
A = np.eye(2) * 3
B = np.eye(3) * 2
np.block([
	[A, np.zeros((2, 3))],
	[np.ones((3,2)), B]
])
	# 合并blocks

a = np.arange(3)
b = np.arange(2, 5)
np.block([a, b, 10])
	# 相当于`np.hstack([a, b, 10])
	# 可以使用单个元素直接做参数

A = np.ones((2, 2), int)
B = 2 * A
np.block([A, B])
	# 相当于`np.hstack([A, B])`
np.block([[A], [B]])
	# 相当于`np.vstack([A, B])`

a = np.array(0)
b = np.array([1])
np.block([a])
np.block([b])
	# 相当于`np.atleast_1d`
np.block([[a]])
np.block([[b]])
	# 相当于`np.atleast_2d`
```

####	`apply_over_axes`

```python
NDA = np.apply_over_axes(
	func(callable(a, axis)),
	a(array-like),
	axes([int])
)
```

####	`apply_over_axis`

```python
NDA = np.apply_over_axis(
	func1d(callable(1DA)),
	axis = int,
	arr = NDA,
	*args,
	**kwargs
)
```

-	说明：对`arr`沿`axis`轴应用`func1d`（即返回NDA shape
	为除该轴的shape）

####	`ndindex`

```python
np.ndindex()
```

