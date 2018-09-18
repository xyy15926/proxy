#	Pandas函数目录

##	“内容结构”变换

###	合并

####	`merge`

```python
DF = pd.merge(
	left(DF),
	right(DF),
	on=col_name/[col_names],
	left_on="left_col_name",
	right_on="right_col_name",
	left_index=False/True,
	right_index=False/true,
	how="Inner"/"outer"/"left"/"right")
```

-	功能：合并操作，类似于sql的*join*操作

-	参数
	-	`on`：merge合并基准列，可以是多个列名称的list，df1、
		df2仅有一列名称相同时可省略，否则返回空DF对象
	-	`left_on`、`right_on`：df1、df2合并基准列名称不同时
		使用
	-	`left_index`、`right_index`：默认为`False`，值为
		`True`时使用索引作为基准进行合并（此时也可以使用
		`df1.join(df2)`）
	-	`how`：合并方式，默认是inner join，参数取值：'outer'
		、'left'、'right'（不知道能不能取'inner'，这个应该
		是默认取值，可以使用）

-	其他
	-	df1、df2有多个列名相同且不全是合并基准列时，返回的
		DF对象的重复列名称会改变

####	`join`

```python
DF = df1.join(
	other(DF/Ser/[DF]),
	on=None/[col_names],
	how="Left/right/outer/inner",
	lsuffix=str,
	rsuffix=str,
	sort=False)
```

-	说明：和其他DF对象进行join操作

-	参数
	-	`other`
		-	参数为Ser时，必须设置field名称
	-	`on`
		-	默认`None`，按照index-on-index进行join
		-	也可以按照col_name进行join
	-	`how`：同上
	-	`lsuffix`：df1重名列使用的后缀
	-	`rsuffix`：df2重名列使用的后缀
	-	`sort`：按照join-key排序

####	`concat`

```python
Df/Ser = pd.concat(
	objs=[Ser, DF, dict],
	axix=0/1/"index"/"columns",
	join="Outer"/"inner",
	join_axes=None/[index],
	ignore_index=False/True,
	keys=None/[]/[()],
	levels=None/[],
	names=None/[index_names],
	verify_integrity=False,
	copy=True)
```

-	说明：以某个轴为方向将多个Series对象或DF对象拼接

-	参数
	-	`objs`
		-	dict作为参数值传递，排序后的keys传递给`keys`
	-	`join`：处理其他轴的方式（其他轴长度、命名不同）
	-	`join_axes`：指定其他轴的index
	-	`ingore_index`：默认`False`，为`True`拼接后的DF的
		Index将为RangeIndex
	-	`keys`：指定构建多级索引最外层Index
	-	`levels`：用于构建多重索引的具体层级，默认从`keys`
		推断
	-	`names`：返回DF对象索引名称
	-	`verify_integrity`：默认`False`，不检查返回DF对象
		是否含有重复index
		`copy`：默认拷贝数据

-	其他
	-	`pd.concat`只涉及拼接方向，而merge只能沿列数增加的
		方向“拼接”

	-	`pd.concat()`时即使有相同的列名称、index序号也不会
		重命名

	-	`pd.concat(axis=1,...)`和
		`pd.merge(left_index=True, right_index=True,...)`的
		作用应该是一样的，只是不会将相同的列名称重命名

	-	`pd.merge`可以指定合并基准列，而`pd.concat`只能按
		Index“合并”，且只能inner join或时outer join

####	`combine_first`

```python
Ser = ser1.combine_first(other(Ser))
Df = df1.combine_first(other(DF))
```

-	说明：和其他DF/Ser进行元素级别的combine，即填充**NULL**
	元素
	-	元素级别
	-	返回对象包含所有的index


###	增、删

####	`drop`

```python
DF/None = df1.drop(
	labels=None/label/[labels],
	axis=0/1/"index"/"columns",
	index=None/index_name/[index_names],
	columns=None/col_name/[col_names],
	level=None/int/level_name,
	inplace=False,
	errors=="Raise/ignore")
```

-	说明：删除df1某轴上的labels

-	参数
	-	`columns`/`index`：替代`axis`+`label`指定需要删除的
		列/行

####	`[new_col_name]`

```python
df1[new_col_name] = list
df1.loc[new_index_name] = list
```

-	说明：添加新列/行
	-	`.iloc[new_index_name]`不可用

####	`append`

```python
DF = df1.append(
	other(DF/Ser/dict/[]),
	ignore_index=False,
	verify_integrity=False)
```

-	说明：将`other`行追加到df1

####	`del`

```python
del df1[col_name]
	# python自身语法直接删除某列
```

##	形、态变换

###	形状

####	`stack`

```python
DF/Ser = df1.stack(
	level=-1/int,
	dropna=True)
```

-	说明：**旋转**`level`级列索引

-	参数
	-	`dropna`：默认剔除返回DF/Ser对象中NULL行

####	`unstack`

```python
DF/Ser = df1.unstack(
	level=-1/int,
	fill_value=None)
```

-	说明：**旋转**`level`级行索引

-	参数
	-	`fill_value`：替换NaN的值

-	其他
	-	两个函数好像都有排序操作，`df1.unstack().stack()`会
		合并层级索引

###	排序

####	`sort_index`

```python
DF = df1.sort_index(
	axis=0/1,
	level=None/int/str,
	ascending=True,
	inplace=False,
	kind="Quicksort"/"mergesort"/"heapsort",
	na_position="First"/"last",
	sort_remaining=True/False)
```

-	说明：按照labels排序

-	参数
	-	`kind`：排序方法
	-	`na_position`：默认将NaN放在开头
		-	对多重索引无效
	-	`sort_remaining`：默认依次对剩余层级排序

####	`sort_value`

```python
DF = df1.sort_value(
	by(label/[labels]),
	axis=0/1,
	ascending=True/False,
	inplace=False,
	kind="Quicksort"/"mergesort"/"heapsort",
	na_position="Last"/"first")
```

-	说明：依某labels（行、列）值排列

####	`rank`

```python
DF = df1.rank(
	axis=0/1,
	method="Average"/"min"/"max"/"first"/"dense",
	numeric_only=None/True/False,
	na_option="Keep"/"top"/"bottom",
	ascending=True,
	pct=False)
```

-	说明：沿轴计算数值型数据的rank
	-	rank值相同者取rank值平均

-	参数
	-	`method`
		-	`average`：组（延轴分组）rank平均
		-	`min`/`max`：组内最小/大
		-	`first`：label出现顺序优先
		-	`dense`：类似`min`，但是rank值总是增加1
	-	`numeric_only`：默认不考虑**含有**非数值型数据label
		-	但是文档上确实写默认值为`None`
	-	`na_option`
		-	`keep`：NaN值rank值不变
		-	`top`：NaN作为“小”值
		-	`bottom`：NaN作为“大”值
	-	`ascending`：默认小值rank值小
	-	`pct`：默认不计算rank值占比

####	`take`

```python
DF = df1.take(
	indices([indice]),
	axis=0/1,
	covert=True,
	is_copy=True,
	**kwargs)
```

-	说明：按照indices对应的labels顺序返回DF

-	参数
	-	`indices`：指明返回indices、及其顺序
		-	indices是指行数，不是labels
		-	可以为负值indice，类似list
	-	`convert`：是否处理负值indice（将废除，均处理）
	-	`is_copy`：默认创建副本返回

####	`reindex`

```python
DF = df1.reindex(
	labels=None/[labels]/Index,
	index=None/[labels]/Index,
	columns=None/[labels],
	axis=0/1/"index"/"columns",
	method=None/"backfill"/"bfill"/"pad"/"ffill"/"nearest",
	copy=True/False,
	level=None/int/level_name,
	fill_value=NaN/scalar,
	limit=None/limit,
	tolerance=scalar/array-like)
```

-	说明：将DF对象转换有其他Index的对象
	-	可以包含之前没有的labels
	-	类似于labels版本`.take`，是**选择**不是重命名

-	参数
	-	`labels`：新Index，配合`axis`决定替换轴
	-	`index`/`columns`：两根轴的Index
	-	`method`：新labels值填补方法
		-	用于填补的值不一定会出现在返回DF对象中，可能是
			使用原DF对象中未被选择labels
	-	`copy`：默认拷贝副本
	-	`fill_value`：新labels值填补值
	-	`limit`：允许最长连续使用`method`方法填补值
	-	`tolerance`：使用`method`方法填补新labels值时，用于
		填补labels和新labels的最大差距
		-	超过`tolerance`则为默认NaN

##	数据处理

###	简单统计

```python

Ser = df1.sum(
	level=None/0/1,
	axis=0/1)

Ser = df1.mean(
	axis=0/1)

Ser = df1.std(
	axis=0/1)

DF = df1.describe()

DF = df1.corr()

DF = df1.cov()

float = ser1.corr(
	ser1)

Ser = df1.corwith(
	other(DF/Ser),
	axis=0/1,
	drop=False/True)
	# `other`为DF对象时计算相同名称相关系数
```

####	`value_count`

```python
Ser = pd.value_counts(
	values(ndarray(1d)),
	sort=True/False,
	ascending=False/True,
	normalize=False/True,
	bins=None/int/[num],
	dropna=True/False)
```

-	说明：计算hisgram

-	参数
	-	`sort`：默认按值排序
	-	`normalize`：默认不正则化（计算相对histgram）
	-	`bins`：hisgrams划分bins
		-	默认每个值划分为一个bins
		-	给出`int`时表示数目，`(min, max]`均等分
		-	给出`[num]`计算列表内数量（范围外不考虑）
	-	`dropna`：默认不计算NaN值个数

####	`quantile`

```python
Ser/DF = df1.quantile(
	q=0.5/float/[float],
	axis=0/1/"index"/"columns",
	numeric_only=True,
	interpolation="linear")
```

-	说明：计算`q`分位数

-	参数
	-	`q`：分位，可以是列表，计算多个分位数
	-	`interpolation`：分位数计算方式（分位数位i、j间）
		-	`linear`：`i+(j-i)*fraction`（线性回归）
		-	`low`：`i`
		-	`high`：`i+1`
		-	`nearest`：`i`、`i+1`中近者
		-	`midpoint`：(`i`+`j`)/2

##	元素级

###	Apply

####	`apply`

```python
DF/Ser = df1.apply(
	func(func/{label: func}),
	axis=0/1/"index"/"columns",
	broadcast=False,
	raw=False,
	reduce=None/True/False,
	args=(),
	**kwargs)
```

-	说明：对`df1`沿轴方向labels应用func
	-	可以用于DFGB对象
		-	为聚合函数时返回DF对象Index为groupby键，类似于
			`agg`
		-	非聚合函数时返回DF对象为原Index，类似于
			`transform`，但包含用于groupby的label
		-	但是此时其他参数无法使用，`func`也仅能为单一
			function，而`agg`可以使用各种

-	参数
	-	`broadcast`：仅对aggregation（聚合）函数，默认不保持
		原shape
		-	0.23.0 deprecated
	-	`raw`：默认不是将label作为ndarray对象传递，而是保持
		Series对象
		-	如果函数处理nadarry对象，`True`性能更好
	-	`reduce`：默认根据函数判断是否返回DF
		-	`False`：尽量返回DF对象
		-	0.23.0 deprecated
	-	`result_type`
		-	`expand`：返回DF对象
		-	`reduce`：尽量返回Ser对象
		-	`broadcast`：保持原shape返回
		-	0.23.0 new
	-	`args`：传递给`func`的VAR_POSITIONs
	-	`kwargs`：传递给`func`的VAR_KEYWORDs

####	`agg`

```python
DF = df1.agg(
	func(callable/"func_name"(str)/{label:func}/[],
	axis=0/1/"index"/"columns",
	*args,
	**kwargs)
```

-	说明：聚合
	-	可以用于DFGB，必然返回DF，因此要求函数**结果**必须
		聚合
		-	如果分组结果都只有单个label，函数可以非聚合
		-	如果分结果中由多label分组，函数必须聚合

-	参数
	-	`func`
		-	dict键应为column labels，值为function

####	`transform`

```python
DF = df1/dfgb1.transform(
	func(callable/"func_name"(str)/dict/[]),
	*args,
	**kwargs)
```

-	说明：返回应用`func`处理后DF
	-	应用于DF对象时，等同于`axis=0`的`agg`、`apply`
	-	应用于DFGB对象时，无价值，和应用于原DF对象结果一样，
		仅剔除groupby label

####	`replace`

```python
DF = df1.replace(
	to_replace=None/str/regex/num/[]/{}/Series,
	value=None/str/num/[]/{},
	inplace=False,
	limit=None/int,
	regex=False,
	method="pad"/"ffill"/"bfill"/"backfill")
```

-	说明：替换

-	参数
	-	`to_replace`：**被替换**对象
		-	str/regex/num：匹配str/regex/num的值被替换为`value`
		-	[ ]
			-	`value`也为list时，长度必须相同
			-	str元素可看作regex
		-	{}
			-	nested dict顶层键匹配列label，然后应用对应
				子dict进行匹配（此时包含`value`功能）
			-	非顶层键可以为regex
		-	None：此时`regex`必为str/[]/{}/Ser
	-	`value`：**替换**值
		-	str/num：替换值
		-	{}：键匹配列label然后替换值，无匹配保持原样
		-	[ ]：长度必须和`to_replace`相同
	-	`limit`：允许最长连续bfill、ffill替换
	-	`regex`
		-	为`True`时，`to_replace`、`value`将作为regex
		-	代替`to_replace`作regex替换
	-	`method`：`to_replace`为list时替换方法

-	说明
	-	`to_replace`为{regex:str}时，只替换DF中str匹配的部分
		，如果需要替换整个str，需要`^.*str.*$`匹配整个str
	-	但为{regex:int}时，不需要`^.*str.*$`也会匹配整个str
		并替换为int

###	选择

####	`where`

```python
df = df1.where(
	cond(df(bool)/callable/[]),
	other=num/df/callable,
	inplace=false,
	axis=none,
	level=none,
	errors="raise",
	try_cast=False,
	raise_on_error=None)
```

-	说明：mask `True`，替换`False`为`other`值，默认（`other`
	中无法找到对应值）NaN
	-	`cond`、`other`是按照`[index][col]`对应位置，而不是
		打印位置

-	参数
	-	`cond`
		-	DF(bool)：`True`保持原值不变，`False`从`other`
			替换
		-	callable：应用在`df1`上，返回DF(bool)，不应改变
			`df
		-	`cond`不提供位置视为被替换1`
	-	`other`
		-	num：替换为num
		-	DF：替换值
		-	callable：应用在`df1`上，返回DF用于替换
	-	`axis`：alignment axis if needed
	-	`level`：alignemnt level if needed
	-	`errors`
		-	`raise`：允许raise exceptions
		-	`ignore`：suppress exceptions，错误时返回原对象
	-	`try_cast`：默认不尝试将结果cast为原输入

####	`mask`

```python
DF = df1.mask(
	cond(DF(bool)/callable/[]),
	other=num/DF/callable,
	inplace=False,
	axis=None,
	level=None,
	errors="raise",
	try_cast=False,
	raise_on_error=None)
```

-	说明：`True`、`False` mask规则同`where`相反，其他同

###	类型转换

####	`to_numeric`

```python

Ser = pd.to_numeric(
	arg([]/()/ndarray(1d)/Ser),
	errors="Raise"/"ingore"/"coerce",
	downcast=None/"integer"/"signed"/"unsigned"/"float")
```

-	说明：转换为numeric类型

-	参数
	-	`errors`
		-	`raise`：无效值将raise exception
		-	`coerce`：无效值设为NaN
		-	`ignore`：无效值保持原样
	-	`downcast`：根据参数downcast值为“最小”类型
		-	downcast过程中exception不受`errors`影响

####	`to_datetime`

```python
Ser = pd.to_datetime(
	arg(int/float/str/datetime/[]/()/ndarray(1d)/Ser/DF,
	error="Raise"/"ignore"/"coerce",
	dayfirst=False,
	yearfirst=False,
	utc=None/True,
	box=True/False,
	format=None/str,
	exact=True,
	unit="ns"/"D"/"s"/"ms"/"us"/"ns",
	infer_datatime_format=False,
	origin="unit")
```

-	说明：转换为datetime

-	参数
	-	`dayfirst`：处理类"10/11/12"，设置day在前
	-	`yearfirst`：处理类"10/11/12"，设置year在前
	-	`utc`：默认不返回UTC DatatimeIndex
	-	`box`
		-	`True`：返回DatetimeIndex
		-	`False`：返回ndarray值
	-	`format`：parse格式，如"%d/%m/%y %H:%M:%S"
	-	`exact`：默认要求`arg`精确匹配`format`
	-	`unit`：`arg`传递数值时作为单位
	-	`infer_datatime_format`：在`format`为`None`时，尝试
		猜测格式，并选择最快的方式parse，耗时5~10倍
	-	`origin`：决定参考（起始）时间
		-	`unix`：起始时间：1970-01-01
		-	`julian`：4714 BC 1.1（此时`unit`必须为`D`）

####	`infer_objects`

```python
DF = df1.infer_objects()
```
-	说明：soft转换数据类型，无法转换保持原样

####	`astype`

```python
DF = df.astype(
	dtype(dtype/{col_name:dtype}),
	copy=True,
	errors="raise"/"ingore",
	**kwargs)
```

-	说明：强制转换为`dtype`类型

-	参数
	-	`copy`：默认返回拷贝
	-	`kwargs`：传递给构造函数的参数

###	Ser

####	`map`

```python
Ser = ser1.map(
	arg={}/Ser/callable,
	na_action=None/"ignore")
```

-	说明：对Ser中元素进行映射
	-	**map**对无法**配置**（dict）返回None而不是保持不变

-	参数
	-	`arg`
		-	{}：对Ser中值根据键值对映射
		-	callable：对元素**应用**callable
	-	`no_action`：默认对NA也处理，否则忽略

-	其他
	-	好像默认会应用类似于`apply(convert_type=True)`，如
		直接取值是`np.float64`类型，传给函数就变成了`float`
		类型

####	`apply`

```python
Ser = ser1.apply(
	func(callable/{}/[]),
	convert_type=True/False,
	args=(),
	**kwargs)
```

-	说明：在Ser上应用`func`

-	参数
	-	`func`
		-	{}：返回**多重索引**Ser，键作为顶层索引，不是对
			不同值应用不同方法
		-	[ ]：返回DF对象，列labels根据list元素命令，list
			元素不能聚合、非聚合混合

	-	`convert_type`：默认转换为合适的类型，否则设为
		`dtype=object`

####	`agg`

```python
Ser = ser1.agg(
	func(callable/{}/[]),
	axis=0,
	args=(),
	**kwargs)
```

-	说明：好像和`apply`完全相同，只是参数不同，但`axis`没用

##	分组

####	`cut`

```python
Ser = pd.cut(
	x(array-like),
	bins(int/[num]/IntervalIndex),
	right=True,
	labels=None/[],
	retbins=False,
	precision=3/int,
	include_lowest=False)
```

-	说明：返回各个元素所属区间，同时也是对应indice

-	参数
	-	`bins`：左开右闭
		-	int：将`x`等分（事实上为了包含最小值，`bins`
			左边界会扩展.1%）
		-	[num]：定义`bins`边界
	-	`right`：默认包含right-most边界
	-	`labels`：指定生成bins的labels
	-	`retbins`：默认不返回bins，设为`True`将返回tuple
	-	`precision`：bins labels显示精度
	-	`include_lowest`：第一个bin是否应该包括最小值
		-	应该仅对`bins=[]`有效
		-	设为`True`仍然为左开右闭区间，但是左边界会小于
			`bins=[]`中的最小值

####	`qcut`

```python
Ser = pd.qcut(
	x(array-like),
	q=int/quantile_list,
	labels=None,
	retbins=False,
	precision=3,
	duplicates="Raise"/"drop")
```

-	说明：将`x`按照`q`划分分位数后进组（即按照数量分组）

-	参数
	-	`q`
		-	int：划分int个等分位数
		-	[num]：将`x`元素按照给出**分位数**分组
	-	`duplicates`
		-	`raise`：bins边缘不唯一raise exception
		-	`drop`：bins边缘不唯一则丢弃不唯一值

####	`groupby`

```python
DFGB = df1.groupby(
	by(col_name),
	axis=0,
	level=None,
	as_index=True,
	sort=True,
	group_keys=True,
	squeeze=False,
	**kwargs)
```

-	说明：根据`by`对DF分组

-	参数
	-	`by`
		-	func：应用于DF的Index上，结果作分组依据
		-	dict/Ser：对Index作映射，映射结果作分组依据
		-	array-like：其值依顺序标记DF行，据此分组，因此
			要求长度必须和DF行数相同
		-	col_name/[col_name]：根据col_name分组
	-	`as_index`：默认使用分组依据作为分组Index（labels）
	-	`sort`：默认组内各保持原DF顺序，可以关闭获得更好性能
	-	`group_keys`：默认在calling apply时，添加group键用于
		标识各组
	-	`squeeze`：为`True`时尽可能减少返回值维度，否则返回
		consistent type

##	Index

###	Index值

####	`pivot`

```python
DF = df1.pivot(
	index=None/str,
	columns=None/str,
	values=None/str)
```

-	说明：根据某列值reshape数据（创建一个**枢**）

-	参数
	-	`index`：用作Index的列名，默认Index
	-	`columns`：用作Column的列名
	-	`values`：填充进新DF对象的列名，默认所有剩余所有列
		（列索引为层级索引）

-	其他
	-	但是如果选取的两列元素两两组合有重复回报错

####	`set_index`

```python
DF = df1.set_index(
	keys(col_name/[ ]/[col_names,[ ]]),
	drop=True,
	append=False,
	inplace=False,
	verify_integrity=False)
```

-	说明：用一列/多列作为新DF的Index（row labels）

-	参数
	-	`keys`：列名，列名列表
	-	`drop`：默认删除用作Index的列
	-	`append`：默认不保留原Index（不以**添加**方式设置
		Index）
	-	`verify_integrity`：默认不检查新Index是否唯一，直至
		必要的时候

####	`reset_index`

```python
DF = df1.reset_index(
	level=None/int/str/[int, str],
	drop=False,
	inplace=False,
	col_level=0/int/str,
	col_fill='')
```

-	说明：

-	参数
	-	`level`：默认所有层级
	-	`drop`：默认将Index作为一列添加
	-	`col_level`：Index作为列添加进的列索引层次，默认
		最高层
	-	`col_fill`：对于多级索引，Index作为列添加时其他层级
		的命名，默认xtts 

-	其他
	-	Index作为新列的名称默认为"index"/"level_0"、
		"level_1"等

###	Index属性

####	`swaplevel`

```python
DF = df.swaplevel(
	i=-2/int/str,
	j=-1/int/str,
	axis=0)
```

-	说明：交换i、j两层索引

####	`rename`

```python
DF = df.rename(
	mapper=None/dict/func,
	index=None/dict/func,
	columns=None/dict/func,
	axis=0/1/"index"/"columns",
	copy=True,
	inplace=False,
	level=None/int/str)
```

-	说明：修改labels（行、列名）

-	参数
	-	`mapper`：labels的重命名规则
	-	`index`/`columns`：行、列labels重命名规则
		-	`mapper`+`axis`
		-	`index`
		-	`columns`
	-	`copy`：默认复制数据
		-	`inplace`为`True`时，此参数无效
	-	`level`：默认重命名所有level

####	`add_prefix`

```python
DF = df.add_prefix(
	prefix(str))
```

-	说明：为列labels添加前缀

##	特殊值

###	重复

####	`unique`

```python
bool = ser1.is_unique
	# 无重复元素**属性**为`True`
Ser = ser1.unique()
	# 返回非重复元素
```

####	`duplicated`

```python
Ser(bool) = df1.duplicated(
	subset=None/label/[labels],
	keep="First"/"last"/False)
```

-	说明：返回标识“副本（元素相同）”行的Ser(bool)

-	参数
	-	`subset`：默认检查所有列，否则检查指定列
	-	`keep`
		-	`first`：重复者除第一个外标记为`True`
		-	`last`：重复者除最后一个外标记为`True`
		-	False：重复者均标记为`True`

```python
DF = df1.drop_duplicates(
	subset=None/label/[labels],
	keep="First"/"last"/False,
	inplace=False)
```

###	空

####	`null`

```python
DF(bool) = df1.isnull()
DF(bool) = df1.notnull()
```

####	`dropna`

```python
DF = df1.dropna(
	axis=0/1,
	how="Any"/"all",
	thresh=None/int,
	subset=None/label/[labels],
	inplace=False)
```

-	说明：剔除空labels（及其数据）

-	参数
	-	`how`：默认any NaN值存在剔除label
	-	`thresh`：剔除label的NaN值阈值
	-	`subset`：指定考虑的labels

####	`fillna`

```python
DF = df1.fillna(
	value=None/scalar/dict/Ser/DF,
	method=None/"backfill"/"bfill"/"pad"/"ffill",
	axis=0/1/"index"/"columns",
	inplace=False,
	limit=None/int,
	downcast=None/dict,
	**kwargs)
```

-	说明：填补NaN值

-	参数
	-	`value`：用于填补NaN的值，dict-like时无法找到的NaN
		不被填补
	-	`method`
		-	"pad"/"ffill"：使用last（前一个）值填补
		-	"bfill"/"backfill"：使用next（后一个）值填补
	-	`limit`：填补允许的最大连续NaN值，超过部分保留NaN
	-	`downcast`：数据类型精度降级dict，或者`"infer"`由
		自行推断

###	其他

####	`any`

```python
Ser = df1(bool).any(
	axis=0/1/"index"/"columns",
	bool_only=None,
	skipna=True,
	level=None/int/str,
	**kwargs)
```

-	说明：label（行、列）对应值存在真返回True

-	参数
	-	`skipna`：默认skip NA，如果全label为NA则结果为NA
	-	`level`：默认考虑整个索引
	-	`bool_only`：默认将所有转换为bool值再做判断

