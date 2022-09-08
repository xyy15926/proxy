
title: Pandas Briefs
categories:
  - Python
  - Pandas
tags:
  - Python
  - Data Analysis
  - Data Visualization
  - Numpy Array
  - Pandas
date: 2022-07-26 11:28:10
updated: 2022-09-07 18:21:46
toc: true
mathjax: true
description: 
---

##	`DataFrame`、`Series`

###	属性、底层数据

|`DF` 方法、属性|描述|返回值|说明|
|-----|-----|-----|-----|
|`index`|行索引| | |
|`columns`|列索引| | |
|`dtypes`|数据类型| | |按列|
|`info([verbose, buf, max_cols,...])`|简介| | |
|`values`|*Numpy* 数组|`np.ndarray`| |
|`axes`|行、列索引|列表| |
|`ndim`|维度（轴向数）|`uint`| |
|`size`|元素数量|`uint`| |
|`shape`|形状|`tuple[int, int]`| |
|`memeory_usage([index,deep])`|内存占用| |按列、`bytes` 计|
|`empty`|是否为空| |`bool`|
|`set_flags(*[,copy,...])`|设置标志| | |
|`keys()`|获取信息轴| |即列轴向|

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#attributes-and-underlying-data>

##	索引访问

###	索引抽取、赋值

|`DF` 访问|描述|返回值|说明|
|-----|-----|-----|-----|
|`loc([axis])[]`|行优先、基于标签的索引、赋值| | |
|`iloc([axis])[]`|行优先、基于位置的索引| | |
|`at[]`|行优先、基于标签的索引| |`D=0`| |
|`iat[]`|行优先、基于位置的索引|`D=0`| |
|`[]`|基于标签的索引、赋值| | |
|`get(key[,default])`|基于列、标签的索引| |类似 `[]`，但可设置默认值|

> - 数据抽取：<https://www.pypandas.cn/docs/user_guide/indexing.html>
> - 数据抽取：<https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html>

####	`.loc`、`.iloc`

-	`.loc[]`、`.iloc[]` 分别为基于标签、位置的列优先抽取
	-	缺省均为行优先抽取，但可传参 `axis` 指定轴向
	-	均支持在 `[]` 内使用元组（即 `,` 分隔）表示多轴向的抽取
		-	不同方式的索引值可以组合，抽取不同轴向
		-	可将元组视为层级深入、列表视为同层筛选
		-	注意：多层级索引也以元组作为索引值，且优先级更高
	-	`.loc`、`.iloc` 被用于设置（赋值） `DataFrame` 时，会提前对齐所有 `Axes`
		-	即，赋值时无需对齐 `Axes` 顺序
		-	也即，无法通过 `.loc`、`iloc` 交换列值
			-	可通过 `.to_numpy` 将转换右值类型，避免对齐
		-	其中，`.loc` 索引不存在标签赋值时，将扩展主体

-	`.loc[]`、`.iloc[]` 均支持以下 5 种方式作为索引值
	-	单个索引值：要求索引值与索引类型兼容
		-	`.loc[]` 基于标签，则支持可转换为索引类型的字符串等
		-	`.iloc[]` 基于位置，仅支持可转换为索引类型的数值类型
	-	列表、数组：同样的要求组成元素应与行索引（数据类型）兼容
		-	此处不允许使用元组，否则被识别为多轴向抽取、多层级索引
		-	若，列表中包含不存在索引值，将 `raise KeyError`、`raise IndexError`
	-	切片对象：同样的要求组成元素应与行索引（数据类型）兼容
		-	按行索引顺序确定切片结果
			-	标签切片包含首、尾，位置切片不包含尾
			-	切片对象越界部分被忽略
		-	切片需要排序性，`.loc` 支持标签切片，则索引
			-	已排序：行为可类似整形切片
			-	未排序：切片端点需在索引中精确、独立匹配，否则将 `raise KeyError`
			-	另，`MultiIndex` 上切片只能在 `lexsort_depth` 指定的层级范围内
	-	`bool` 数组
		-	`NA` 等值被视为 `False`
		-	`.loc[]` 支持带标签的 `bool` 数组（`pd.Series`）
		-	`.iloc[]` 支持仅带位置的 `bool` 数组（`np.ndarray`）
	-	`callable` 对象
		-	参数：`Series`、`DataFrame` 整体
		-	返回值：前述 4 种索引值

```python
df = pd.DataFrame(np.random.rand(5,4),
	columns = list("ABCD"),
	index = pd.data_range("20220101", periods=5))
df.loc["20220101": "20220103"]					# 可转换为标签的字符串作为切片
df.loc(axis=1)["A": "C"]						# 切换轴向
```

####	`[]`、`.`

-	`[]` 行为类似 `.loc[]`，但
	-	不支持在 `[]` 内使用 `,` 分隔多个轴向的抽取
	-	一般抽取列（列优先），但以下情况抽取行
		-	切片对象：同时支持标签、位置（仅在 `Int64Index` 情况下优先）
		-	`bool` 数组：同时支持 `pd.Series`、`np.ndarray`
	-	另，可通过 `.` 属性访问方式类似 `[]` 抽取数据，但要求索引值
		-	为有效标识符
		-	不与现有方法、属性冲突
	-	另，支持传入 `Series` 重排列索引

> - `.loc`、`[]` 用于设置不存在的标签时，将执行新增数据

####	说明事项

-	混合位置、标签索引需将位置、标签互相转换，再利用 `.iloc`、`.loc`
	-	`pd.Index[<POS>]`：位置转换为标签
	-	`pd.Index.get_loc(<LABEL>)`：根据标签获取位置
	-	`pd.Index.get_indexer([<LABEL>])`：根据标签获取位置

-	尽量在一步操作中完成全部索引操作，避免使用链式索引
	-	链式索引：连续索引操作
		-	显示链式索引即无显式中间变量的连续索引操作
		-	隐式链式索引即将中间索引结果显示赋给变量
	-	无法保证链式索引中每步返回结果为视图、或副本（取决于数组内存布局），会导致意外情况
		-	赋值时可能意外地修改副本
			-	默认地，*Pandas* 会打印 `SettingWithCopyWarning`
			-	可设置 `model.chained_assignment` 选项控制 *Pandas* 处理逻辑
	-	链式索引效率更低

-	`.loc[]` 中元组作为索引时的解析优先级说明
	-	若元组元素非 *hashable*（如：列表），仅视为不同轴向抽取
		-	但，主体为 `Series` 时，亦可视为多层级索引
	-	若元组元素可 *hashable*，优先视为多层级索引
	-	多轴向抽取、多层级索引不支持解包混合
		-	即，`[s1, s2, s3]` 无法被解释为 `s1, s2` 作为行多层级索引、`s3` 作为列索引
		-	即，需用 `()` 分隔、包装为 2 元组，以混合多轴向、多层级索引
	-	即，多层级、多轴项索引需一定包装避免歧义
		-	2 元组包装：`,` + `:`
		-	`pd.IndexSlice` 包装：界定某轴向索引
		-	`.loc(axis=<AX>)`：参数指定轴向

###	访问

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`take(indices[,axis,is_copy,...])`|基于位置索引| |不支持 `bool` 索引，少量索引时速度更快|
|`xs(key[,axis,level,drop_level])`|从特定多级索引、层级抽取|同层,低层|不可用于赋值|
|`head([n])`|首 `n` 行|`R=n`| |
|`tail([n])`|尾 `n` 行|`R=n`| |
|`sample([n,frac,replace,...])`|抽样|`R=n`| |

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#indexing-iteration>

###	赋值

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`insert(loc,column,value[,...]`|插入列|`C+1`| |
|`assign(**kwargs)`|关键字参数插入列|`C+len(kwargs)`| |

###	迭代

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`items()`|迭代列名、列| | |
|`iteritems()`|同 `items()`| | |
|`iterrows()`|迭代行|`<IDX>, <SER>`元组| |
|`itertuples([index,name])`|以命名元组方式迭代行| | |
|`keys()`|获取 *Info Axis*| |即列轴|
|`pop(item)`|迭代、删除列| | |

> - `DF` 缺省迭代均迭代 *info axis*（即列）
> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#indexing-iteration>

###	数据删除

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`drop([labels,axis,index,...])`|按标签、沿轴向丢弃|`D`| |

##	数据比较、筛选

###	算数比较

|算数比较方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`lt(other[,axis,level])`|逐元素比较|广播后 `bool` 形状，下同|`axis`、`level` 用于对齐广播，下同|
|`gt(other[,axis,level])`| | | |
|`le(other[,axis,level])`| | | |
|`ge(other[,axis,level])`| | | |
|`ne(other[,axis,level])`| | | |
|`eq(other[,axis,level])`| | | |
|`>`、`<`、`>=`、`<=`、`==`|逐元素比较|同形状 `bool`| |

###	元素操作

####	逐元素操作

|逐元素|描述|返回值|说明|
|-----|-----|-----|-----|
|`abs()`| | | |
|`clip([lower,upper,axis,inplace])`|按阈值截断|同形状| |
|`cummax([axis,skipna])`|累计最大值| | |
|`cummin([axis,skipna])`|累计最小值| | |
|`cumprod([axis,skipna])`|累乘| | |
|`cumsum([axis,skipna])`|累加| | |
|`diff([periods,axis])`|差分| | |
|`round([decimals])`|精度| | |

####	沿轴聚集

|沿轴聚集|描述|返回值|说明|
|-----|-----|-----|-----|
|`all([axis,bool_only,skipna,level])`|全真|`bool` 行、列| |
|`any([axis,bool_only,skipna,level])`|存在真|`bool` 行、列| |
|`count([axis,level,numeric_only])`|计算非 `NA` 数量| |
|`nunique([axis,dropna])`|计算不重复元素数量| | |
|`describe([percentile, include,...])`|按列统计指标| | |
|`kurt([axis,skipna,level,...])`|无偏峰度| | |
|`kurtosis([axis,skipna,level,...])`|同上| | |
|`sem([axis,skipna,level,ddof,...])`|标准差| | |
|`skew([axis,skipna,level,ddof,...])`|标准差| | |
|`sem([axis,skipna,level,ddof,...])`|无偏标准误| |多次抽样均值波动，可用 `=<STD>/sqrt(n)` 估算|
|`std([axis,skipna,level,ddof,...])`|无偏标准差| | |
|`var([axis,skipna,level,ddof,...])`|无偏方差| | |
|`mad([axis,skipna,level])`|绝对值偏差| | |
|`max([axis,skipna,level,...])`|最大值| | |
|`mean([axis,skipna,level,...])`|均值| | |
|`median([axis,skipna,level,...])`|中位数| | |
|`min([axis,skipna,level,...])`|最小值| | |
|`mode([axis,skipna,level])`|众数| |可能有多个|
|`pct_change([periods,fill_method,...])`|变化率| | |
|`sum([axis,skipna,level,...])`|累加| | |
|`prod([axis,skipna,level,...])`|累乘| | |
|`product([axis,skipna,level,...])`|同上| | |
|`quantile([q,axis,skipna,level,...])`|分位数| | |
|`rank([axis,skipna,level,...])`|秩（排序）| | |
|`nunique([axis,dropna])`|唯一值数量| | |

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats>

####	跨轴操作

|跨轴操作|描述|返回值|说明|
|-----|-----|-----|-----|
|`corr([method,min_periods])`|列配对相关系数|`C * C`| |
|`corrwith(other[,axis,drop,method])`|列配对相关系数|`C1 * C2`| |
|`eval(expr[,inplace])`|按表达式执行| | |
|`value_counts([subset,normalize,...])`|计算不重复行数量|`C+1`| |

###	特殊筛选

|数据筛选方法|描述|入参|返回值|说明|
|-----|-----|-----|-----|-----|
|`isin(values)`|元素是否存在|同形状 `bool`| |
|`nlargest(n,columns[,keep])`|按列值最大 `n`|`R=n`| |
|`nsmallest(n,columns[,keep])`|按列值最小 `n`|`R=n`| |
|`query(expr[,inplace])`|基于字符串表达式的索引| |大数据量时速度更快|
|`select_dtypes([inlcude, exclude])`|按数据类型选列|`c-`|支持 *Numpy* 泛型类作参数|

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#reindexing-selection-label-manipulation>

###	索引筛选

|索引筛选方法|描述|入参|返回值|说明|
|-----|-----|-----|-----|-----|
|`first_valid_index()`|首个非 `NA` 索引|`D=0`| |
|`last_valid_index()`|末个非 `NA` 索引|`D=0`| |
|`idxmax([axis,skipna])`|沿轴向各最大标签|`D=1`| |
|`idxmin([axis,skipna])`|沿轴向各最小标签|`D=1`| |
|`filter([items,like,regex,axis])`|筛选满足条件索引的数据|`D`| |
|`truncate([before,after,axis,copy])`|沿轴向按标签截断|`D`| |
|`reindex([labels,index,columns,...])`|重设索引|`D`|标签可不存在，此时可能改变索引类型、数据类型，重复索引项报错|
|`reindex_like(other[,method,...])`|参考给定对象重设索引|`D(other)`|同上|
|`asof(where[,subset])`|（较小）最接近项|`D-1`|索引项无重复、无匹配置 `nan`|

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#reindexing-selection-label-manipulation>

##	计算、更新

###	算数运算

|算数计算方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`add(other[,axis,level,fill_value])`| |广播后形状，下同|`axis`、`level` 用于对齐广播，下同|
|`sub(other[,axis,level,fill_value])`| | | |
|`mul(other[,axis,level,fill_value])`| | | |
|`div(other[,axis,level,fill_value])`| | | |
|`truediv(other[,axis,level,...])`| | | |
|`floordiv(other[,axis,level,...])`| | | |
|`mod(other[,axis,level,fill_value])`| | | |
|`pow(other[,axis,level,fill_value])`| | | |
|`dot(other[,axis,level,fill_value])`| | | |
|`radd(other[,axis,level,fill_value])`| |广播后形状，下同|`axis`、`level` 用于对齐广播，下同|
|`rsub(other[,axis,level,fill_value])`| | | |
|`rmul(other[,axis,level,fill_value])`| | | |
|`rdiv(other[,axis,level,fill_value])`| | | |
|`rtruediv(other[,axis,level,...])`| | | |
|`rfloordiv(other[,axis,level,...])`| | | |
|`rmod(other[,axis,level,fill_value])`| | | |
|`rpow(other[,axis,level,fill_value])`| | | |
|`equals(other)`|对应元素均相等|`bool D=0`| |

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#binary-operator-functions>

###	泛用应用

|逐元素|描述|返回值|说明|
|-----|-----|-----|-----|
|`applymap(func[,na_action])`|逐元素应用| | |
|`apply(func[,axis,raw,...])`|沿轴向应用|形状取决于 `func`|`func` 不可为列表|
|`transform(func[,axis])`|沿轴向应用| |`func` 不可为聚集函数|
|`agg([func,axis])`|沿轴向应用|`R/C=len(func) * RET`|`func` 可为列表、非聚集函数|
|`aggregate([func,axis])`|同 `agg`| | |
|`groupby([by,axis,level,...])`|分组| |组可应用上述方法应用聚集、或直接调用聚集函数|
|`rolling(window[,min_periods,...])`|滑动窗口| |窗口同上|
|`expanding([min_periods,axis,...])`|扩展窗口（累计）| |窗口同上|
|`ewm([com,span,halflife,alpha,...])`|指数平滑| |窗口同上|
|`pipe(func,*args,**kwrags)`|链式调用| |等价于 `func(df,...)`|

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#function-application-groupby-window>

###	*Groupby* 应用

|*Groupby* 属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`pd.Grouper(key,level,freq,axis,...)`|创建分组指示器|`Grouper`| |
|`__iter__()`|迭代| | |
|`groups`|分组、标签|`{<GROUP>:[LABELS]}`| |
|`indices`|分组、位序|`{GROUP>:<POS>}`| |
|`get_group(name[,obj])`|获取指定组别|`S`| |
|`apply(func,*args,**kwargs)`|逐组应用，合并结果|`D=1`| |
|`agg(func,*args,**kwargs)`|同 `aggregate`| | |
|`aggregate(func,*args[,engine,...])`|逐组、按列聚集|`R-,C*len(func)`| |
|`transform(func,*args[,engine,...])`|逐组、按列转换|`R,C*len(func)`|结果广播，即同组结果相同|

-	`aggregate`、`transform` 说明
	-	支持关键字传参以设置列名
	-	函数可用 `pd.NamedAgg<"columns","aggfunc">` 封装以仅对特定列应用

> - *Groupby API*：<https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html>

###	值更新

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`where(cond[,other,inplace,...])`|基于 `bool` 真值更新|`S`|对应 `pd.where`、缺省填充空值|
|`mask(cond[,other,inplace,axis,...])`|基于 `bool` 假值更新|`S`|与 `DF.where` 逻辑相反|
|`update(other[,on,how,lsuffix,...])`|更新值|`S`| |
|`compare(other[,join,overwrite,...])`|比较、返回不同|`D`| |
|`replace(to_replace,value,...])`|替换值|`S`| |
|`combine(other,func[,fill_value,...])`|逐列配对调用 `func`| | |

###	类型转换

|`DF` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`astype(dtype[,copy,errors])`|转换类型|`S`| |
|`convert_dtypes()`|转换为新式类型|`S`|以支持 `StringDtype`、`NA` 等|
|`infer_objects()`|推断类型|`S`| |
|`bool()`|转换为 `bool`|`bool S`| |
|`pd.to_numeric(args,errors,downcast)`|转换为数值| | |
|`pd.to_datetime(args,errors,format)`|转换为 `datetime64`| | |
|`pd.to_timedelta(args,errors)`|转换为 `timedelta64`| | |

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#conversion>

###	（顶层）工具函数

|通用工具函数|描述|返回值|说明|
|-----|-----|-----|-----|
|`factorize(value,sort)`|因子化（编码）|编码序列、因子|类似 `np.unique`|
|`unique(value)`|唯一值|数组|根据哈希表确认|
|`cut(x,bins)`|值分箱|分类、箱（需置位 `retbins`）| |
|`qcut(x,q)`|频分箱|分类、箱（需置位 `retbins`）| |
|`get_dummies(data,prefix,drop_first)`|*one-hot* 化（即指示器）|扩列| |
|`eval(expr[,parser,engine,truediv,...])`|按表达式执行| | |

|构造工具函数|描述|返回值|说明|
|-----|-----|-----|-----|
|`date_range([start,end,periods,freq,tz,...])`|创建固定 `freq` 的 `DatatimeIndex`| | |
|`bdate_range([start,end,periods,freq,tz,...])`|同上，仅工作日| | |
|`period_range([start,end,periods,freq,name])`|创建固定 `freq` 的 `PeriodIndex`| | |
|`timedelta_range([start,end,periods,freq,...])`|创建固定 `freq` 的 `TimedeltaIndex`| |缺省日|
|`infer_freq(index[,warn])`|推测 `freq`| | |
|`interval_range([star,end,periods,freq,...])`|创建固定频率的 `IntervalIndex`| | |

> - 顶层通用函数 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/general_functions.html#data-manipulations>

##	特殊值筛选、操作

###	重复值

|数据筛选方法|描述|入参|返回值|说明|
|-----|-----|-----|-----|-----|
|`duplicated`|是否重复值|标签、列表|`bool D=1`| |
|`drop_duplicates([subset,keep,...])`|丢弃重复值|`D`| |

###	缺失值

|方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`isna()`、`isnull()`|为缺失值|`bool S`|有相应顶层函数|
|`notna()`、`notnull()`|非缺失值|`bool S`|有相应顶层函数|
|`fillna([value,method,axis,...])`|填充缺失值|`S`| |
|`ffill([axis,inplace,limit,downcast])`、`pad()`|前值填充|`S`|同 `fillna(method="ffill")`|
|`bfill([axis,inplace,limit,downcast])`、`backfill()`|后值填充|`S`|同 `fillna(method="bfill")`|
|`dropna([axis,how,thresh,...])`|丢弃缺失值|`D`| |
|`interpolate([method,axis,limit,...])`|插值|`S`|部分插值方法依赖 *SciPy*|

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#missing-data-handling>

####	`np.NaN`

-	`NaN` 是 *Pandas* 中默认缺失值标记，被视为缺失值的值包括
	-	`np.nan`：默认缺失值，`np.float64` 类型实例
		-	不可比，即使是自身
	-	`None`：也被被视为缺失值
	-	`pd.NaT`：`datatime64[ns]` 类型缺失值
		-	在 *Numpy* 中可用 `np.datatime64("nat")` 表示
	-	`inf`：需 `mode.use_if_as_na` 选项置位
	-	`pd.NA`：被用于替代 `None`、`np.nan`、`pd.NaT` 统一缺失类型

-	说明（不考虑 `pd.NA`）
	-	大部分因变换出现的缺失均使用 `np.nan` 填充
		-	仅 `datetime64` 类型使用 `pd.NaT` 填充
		-	会导致数值型、`bool` 型数据发生类型转换为浮点型
	-	插入、设置缺失值时，缺失值会被自动转换为相应类型
		-	数值型：`np.nan`
		-	`datetime64` 类型：`np.NaT` 类型
		-	`object`：不自动转换类型
	-	计算过程中，缺失值一般视为 0、被跳过

####	`pd.NA`

-	`pd.NA`：单例类型值，用于标记标量缺失值
	-	目标：跨类型的缺失标记，以替代 `np.nan`、`None`、`pd.NaT`
		-	可避免因 `np.nan` 类型导致的数据类型上调
		-	但，目前仅作为整形、`bool`、`StringDtype` 的缺失值标记
	-	支持行为
		-	算数操作、比较操作：大部分结果返回 `pd.NA`
		-	`|`、`&` 逻辑运算：仅在需要时，结果返回 `pd.NA`
			-	即，仅在 `&` 另一者为 `True` 时，返回 `pd.NA`
			-	但，`or`、`and` 等涉及 `__bool__` 将 `raise TypeError`
		-	实现 *Numpy* 的 `__array_func__` 协议，大部分 *ufunc* 将返回 `pd.NA`

> - `pd.NA` 仍为实验性机制，行为可能会修改
> - 缺失值处理：<https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html>

###	时序索引

####	时序索引操作

|`DF` 时序筛选|描述|返回值|说明|
|-----|-----|-----|-----|
|`shift([periods,freq,axis,...])`|偏移索引|`S`| |
|`asfreq(freq[,method,how,...])`|转换 `freq`|`R+/-`| |
|`resample(rule[,axis,close,...])`|重抽样（再操作）|`R+/-` 丢弃、增加保证所有项在 `freq` 上|
|`to_period([freq,axis,copy])`|转换为 `PeriodIndex`| | |
|`to_timestamp([freq,how,axis,copy])`|转换为 `DatetimeIndex`| | |
|`tz_convert(tz[,axis,level,copy])`|转换时区|`S`| |
|`tz_localize(tz[,axis,level,...])`|本地化时区|`S`| |

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#time-series-related>

####	时序索引

|`DF` 时序操作|描述|返回值|说明|
|-----|-----|-----|-----|
|`at_time(time[,asof,axis])`|在特定时间| |适用 `DatetimeIndex`|
|`between_time(start_time,end_time)`|属于时间区间| |同上|
|`first(offset)`|最初时间段内| |同上|
|`last(offset)`|最后时间段内| |同上|

##	数据形状

###	连接

|`DF` 连接方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`merge(right[,how,on,left_on,...])`|指定键联结|`D`|两主体、联结键 `on` 可指定索引（层级）名|
|`join(other[,on,how,lsuffix,...])`|右仅索引联结|`D`|同置位 `right_index`、可多主体|
|`combine_first(other)`|更新 `NA` 值|`S+`|同形状|两主体、索引连接|
|`pd.concat(objs[,axis,join,ingore_index,...])`|索引连接|`D[+]`|可多主体|
|`pd.merge(left,right[,how,on,left_on,...)`|键连接|`D`|两主体|
|`pd.merge_ordered(left,right[,how,on,left_on,...])`|外连接并排序|`D`|支持分组、空值填充|
|`pd.merge_asof(left,right[,on,left_on,...])`|按距离模糊连接|`D`|仅左连接|

-	`merge` 连接操作中连接键说明
	-	可传递索引名作为连接键，即作为 `on` 系参数实参
	-	除索引连接索引外，连接键层级需相同
	-	不同方法支持的连接键情况
		-	`concat`：左、右索引
		-	`join`（仅有方法版本）：左自定义键、右索引
		-	`merge`：左、右自定义键，缺省自动寻找相同列名作为连接键

> - <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#combining-comparing-joining-merging>
> - *Merge* 数据连接：<https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>

###	变形

|`DF` 变形方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`align(other[,join,axis,level,...])`|沿轴向对齐索引|`(<L_DF>,<R_DF>)`|缺失值置 `NA`|
|`pivot([index,columns,values])`|透视表|`R=len(index)`、`C=len(columns)`|行、列对不可重复|
|`stack([level,dropna])`|堆叠列索引（层级）至行|`C/`|自动排序索引、缺失值置 `NA`|
|`unstack([level,fill_value])`|堆叠行索引（层级）至列|`R/`|自动排序索引|
|`melt([id_vars,var_name,var_vlaue,...])`|除主键外转换为键值对|`C=len(id_vars)+2`| |
|`explode(column[,ignore_index])`|沿行扩展（即行增长）|`C+`|`column` 中元素可迭代|
|`sqeeze([axis])`|去除轴向|`D-1`|若轴向长非 1，将保持不变|
|`transpose(*args[,copy])`|转置|`R=C`、`C=R`| |
|`pd.pivot_table(data,values,index,columns,aggfunc)`|透视表| |支持聚集函数处理重复行列对|
|`pd.melt(frame,id_vars,var_name,var_value)`|除主键外转换为键值对|列数=主键数+2| |
|`pd.wide_to_long(df,stubnames,i,j)`|列名去前缀作为次级行索引 melt| |不建议使用|
|`pd.crosstab(index,columns,values,aggfunc)`|从属性序列创建列联表|`len(index)*len(columns)`| |

-	`pviot` 轴变换操作说明
	-	`pivot`、`pivot_table` 透视表：在记录表中指定主键（主体作轴）、聚集函数，转换为主体表
		-	本质上，即 `groupby + aggragate`
		-	实际上，`set_index` 也可视为单轴透视操作
		-	注意：`pivot` 在行列对重复时 `raise ValueError`，考虑使用 `pd.pivot_table`
	-	`melt`、`wide_to_long`：在主体表中指定主键、属性，转换为记录（属性、值对）表
		-	逻辑上，与 `pivot` 操作互逆
		-	实务中，记录表中“主键”会重复，`pivot` 中聚集操作丢失信息、不可逆
	-	`stack`、`unstack`：轴方向的转换

```python
melted.pivot(<KEYS>, <ATTR_NAME>, <ATTR_VAL>).reset_index()		# 可还原 `melt` 结果
pivoted.reset_index().melt(<KEYS>)								# 可还原 `pivot` 已聚集结果
pivoted.stack().reset_index()									# 同
```

> - *Pivot* 形状变换：<https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html>
> - *Melt, Pivot* 操作解释：<https://stackoverflow.com/questions/41142372/how-to-unmelt-the-pandas-dataframe>
> - *DataFrame API*：<https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#reshaping-sorting-transposing>

##	索引 `Index`

-	`pd.Index` 类（及子类）：有序、不可变多集合，用于索引、对齐
	-	索引数据不可变，但可以设置其元数据
		-	`name`：名称
		-	`levels`：多层索引层级
	-	索引项默认不要求唯一
		-	方便兼容现实中的数据
		-	可通过 `set_flags` 复位 `allow_duplicate_labels` 禁止重复标签
	-	索引可以包含缺失值，但应避免使用缺失值
		-	某些操作会隐式排除缺失值

> - 高级索引：<https://www.pypandas.cn/docs/user_guide/advanced.html>
> - 高级索引：<https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html>
> - 索引 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html>

###	`Index` 属性

|`Index` 方法、属性|描述|返回值|说明|
|-----|-----|-----|-----|
|`values`|索引值|`D=1`| |
|`is_monotonic`|单调性|`bool D=0`| |
|`is_monotonic_increasing`|单调增|同上| |
|`is_monotonic_decreasing`|单调减|同上| |
|`is_unique`|无重复|同上| |
|`has_duplicates`|有重复|同上| |
|`dtype`|索引值数据类型| | |
|`inferred_type`|推测的数据类型| | |
|`is_all_dates`|只包含日期|`bool D=0`| |
|`shape`|形状|`tuple`| |
|`name`|索引名| | |
|`names`|索引各层级名|`FrozonList`| |
|`nbytes`|内存占用| | |
|`ndim`|数据维度| |按定义为 1|
|`size`|元素数量| | |
|`empty`| | | |
|`memory_usage([deep])`|内存占用| | |

###	`Index` 修改、筛选、变换

|`Index` 方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`copy([name,deep,dtype,names])`|拷贝| | |
|`delete(loc)`|按位置删除| | |
|`drop(label[,errors])`|按索引项值删除|`R-`| |
|`drop_duplicates([keep])`|删除重复值|`R-`| |
|`duplicated([keep])`|项是否唯一|`bool S`| |
|`insert(loc,item)`|插入| | |
|`reindex(target[,method,level,...])`|重建索引|`D`| |
|`rename(name[,inplace])`|重命名|`S`| |
|`repeat(repeats[,axis])`|逐元素复制|`len * repeats`| |
|`where(cond[,other])`|按 `bool` 假替换|`S`| |
|`putmask(mask,value)`|按 `bool` 真替换|`S`| |
|`take(indices[,axis,allow_fill,...])`|按位置选取索引项|`len(indices)`| |

|`Index` 缺失值方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`fillna([value,downcast])`|填充缺失值|`S`| |
|`dropna([how])`|丢弃缺失值|`R-`| |
|`isna()`|缺失值|`S`、`bool`| |
|`notna()`|非缺失值|`S`、`bool`| |

|`Index` 变换方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`astype(dtype[,copy])`|转换索引类型|`S`| |
|`item()`|转换为标量| |类似 `sqeeze`，长度大于 1 将保持不变|
|`map(mapper[,na_ation])`|映射索引项|`S`| |
|`ravel([order])`|展平索引|`C=1`| |
|`to_list()`|转换为 `list`| | |
|`to_slice([index,name])`|转换为 `Series`|`S`|缺省索引、值相同|
|`to_frame([index,name])`|转换为 `DF`|`C=1`|缺省索引、值相同|
|`view([cls])`|转换类型| | |

###	`Index` 判断、工具

|`Index` 判断方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`all(*args,**kwargs)`|全真| | |
|`any(*args,**kwargs)`|存在真| | |
|`argmin([axis,skipna])`|最小值位置| | |
|`argmax([axis,skipna])`|最大值位置| | |
|`is_(other)`|类似 `is`，并支持跨视图| | |
|`is_boolean()`|仅包含 `bool`|`bool D=0`| |
|`is_categorical()`|分类数据索引| | |
|`is_floating()`|浮点索引| |
|`is_integer()`|仅包含整型值| | |
|`is_interval()`|区间索引| | |
|`is_numeric()`|仅包含数值| | |
|`is_mixed()`|包含多种类型| | |
|`is_object()`|`object` 类型索引| | |
|`min([axis,skipna])`|最小值| | |
|`max([axis,skipna])`|最大值| | |
|`equals(other)`|各项值相同|`bool D=0`| |
|`identical(other)`|类似 `equals`，并检查元数据| | |

|`Index` 工具方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`factorize([sort,na_sentinel])`|因子化|编码序列、因子序列| |
|`unique([level])`|唯一值| |类似 `factorize`|
|`nounique([dropna])`|重复值数量|`int`| |
|`value_counts([normalize,sort,...])`|各值数量|`D=1`| |

###	`Index` 运算

|`Index` 运算|描述|返回值|说明|
|-----|-----|-----|-----|
|`argsort(*args,**kwargs)`|索引排序|`np.ndarray`|参数被传递给 `ndarray.argsort`|
|`searchedsorted(value[,side,sorter])`|查找插入位置| | |
|`sort_values([return_indexer,...])`|排序|`S`| |
|`append(other)`|合并 `Index`|`R+`| |
|`join(other[,how,level,...])`|连接|`R+/-`| |
|`union(other[,sort])`|`\|` 并| |需转换为公共 `dtype`|
|`intersection(other[,sort])`|`&` 交| | |
|`difference(other[,sort])`|差| | |
|`symmetric_difference(other[,...])`|`^` 对称差| | |

###	`Index` 筛选

|`Index` 筛选|描述|返回值|说明|
|-----|-----|-----|-----|
|`get_loc(key[,method,tolerance])`|查找单个位置|`int`、`slice`、`bool`|索引项可重复|
|`get_indexer(target[,method,limit,...])`|查到多个位置|`int`、`ndarray`|索引项无重复、模糊匹配、无匹配置 `-1`|
|`get_indexer_for(target)`|查找多个位置|`int`、`ndarray`|索引项可重复|
|`get_indexer_non_unique(target)`|查找多个位置|`(loc, No-Exists)`|索引项无重复、标签可不存在|
|`asof(label)`|（较小）最接近索引项|标签、`ndarray`|索引项无重复、无匹配置 `nan`|
|`asof_locs(where,mask)`|（较小）最接近索引项|标签、`ndarray`|`where` 需为 `Index`、索引项无重复、无匹配置 `-1`|
|`get_level_values(level)`|获取索引某层值| | |
|`get_slice_bound(label,side[,kind])`|查找切片边界| |标签需不中断|
|`isin(values[,level])`|是否属于|`bool`| |
|`slice_indexer([start,end,step,kind])`|根据标签创建位置切片|`slice`| |
|`slice_locs([start,end,step,kind])`|根据标签给出位置切片首、尾|`(int,int)`| |
|`get_value(series,key)`| | | |

###	`Index` 子类兼容方法

|子类兼容方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`set_names(names[,level,inplac])`|命名索引|`S`| |
|`droplevel([level])`|丢弃层级|`C-1`| |
|`shift([periods,freq])`|偏移索引项值|`S`|时序索引兼容方法|

> - 此部分方法定义在 `Index` 类中，但只能用于特定索引类型

###	`DF` 修改索引

|`DF` 修改索引方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`add_prefix(prefix)`|列名加前缀| | |
|`add_suffix(suffix)`|列名加后缀| | |
|`set_axis(labels[,axis,inplace])`|标签重设| | |
|`rename([mapper,index,columns,...])`|标签重命名|`S`| |
|`rename_axis([mapper,index,...])`|轴名重命名|`S`|同 `Index.rename`|
|`set_index(keys[,drop,append,...])`|选取列作为行索引|`D`| |
|`reset_index([level,drop,...])`|重置索引|`D`|索引重置为 `RangeIndex`|
|`droplevel(level[,axis])`|移除索引某层级|`S`| |
|`swaplevel([i,j,axis])`|交换索引层级|`S`| |
|`reorder_levels(order[,axis])`|重排索引层级|`S`| |
|`sort_index([axis,level,...])`|排序索引|`S`| |
|`set_flags(allow_duplicate_labels,...)`|设置标志|`D`| |

##	`Index` 子类

###	*Numeric Index*

|*Numeric Index* 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`RangeIndex([start,stop,step,dtype,copy,...])`|*Range* 索引| | |
|`Int64Index([data,dtype,copy,name])`|64 位整形索引| |不可变数组组成的有序、可切片集合|
|`UInt64Index([data,dtype,copy,name])`|64 位非负整形索引| | |
|`Float64Index([data,dtype,copy,name])`|64 位浮点索引| |包含浮点值时自动创建|
|`RangeIndex.start`|起始| | |
|`RangeIndex.stop`|结束| | |
|`RangeIndex.step`|步长| | |
|`RangeIndex.from_range(data[,name,dtype])`|从 `range` 对象创建| | |

-	数值类型索引
	-	`Int64Index`、`Int64Index`、`Float64Index` 之后将被 `Index` 类型统一取代
	-	`RangeIndex` 范围索引：`Int64Index` 索引子类
		-	节省空间、提高效率
		-	缺省索引类型
	-	说明
		-	数值索引中，`.loc` 总是接受等值数值、不等值切片作为索引
		-	但整形索引中，`[]` 不接受浮点切片作为索引

###	*Categorical Index*

|`CategoricalIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`CategoricalIndex([data,cateogries,...])`|基于底层 `Categorical` 的索引| | |
|`codes`|编码序列| | |
|`categories`|因子| | |
|`ordered`|有序| | |
|`rename_categories(new_categories)`|重命名因子|`S`|不影响序列值|
|`reorder_categories(new_categories,ordered)`|重排序因子|`S`| |
|`add_categories(new_categories)`|添加因子|`S`| |
|`remove_categories(removals)`|删除因子| |被删除元素置 `nan`|
|`remove_unused_categories()`|删除未使用因子| | |
|`set_categories(new_categories,ordered,rename)`|设置因子| | |
|`as_ordered(inplace)`|设为有序| | |
|`as_unordered(inplace)`|设为无序| | |
|`map(mapper)`|映射值|`S`| |
|`equals(other)`|各元素均相同|`bool D=0`| |

-	`CategoricalIndex` 分类索引：围绕 `Categorical` 构建的容器
	-	特点
		-	可高效存储、索引具有大量重复元素的索引
		-	其中索引项为 `CategoricalDtype` 类型
	-	说明
		-	抽取、分组等操作依然保留分类索引的全部信息（在数据类型中存储）
		-	`reindex` 将破坏分类索引
		-	包含分类索引的主体变形、比较时要求分类类型相同，否则 `raise TypeError`

###	*Interval Index*

|`IntervalIndex` 类、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`IntervalIndex(data[,closed,dtype,copy,...])`|不可变区间索引| | |
|`from_arrays(left,right[,...])`|端点序列创建| | |
|`from_tuple(data[,closed,...])`|端点对序列创建| | |
|`from_breaks(breaks[,closed,...])`|分隔点序列创建| | |
|`left`|左端点序列|`Index`| |
|`right`|右端点序列|`Index`| |
|`mid`|区间中点|`Index`| |
|`closed`|端点闭合情况|`str`| |
|`length`|长度| | |
|`values`|区间序列|`IntervalArray`| |
|`is_empty`|区间为空| | |
|`is_non_overlappping_monotonic`|严格单调| | |
|`is_overlapping`|区间有重叠| | |
|`set_closed(closed)`|设置端点闭合|`S`| |
|`contains(other)`|检查包含|`bool S`| |
|`overlaps(other)`|检查交集||`bool S`| |
|`to_tuples(na_tuple)`|转换为元组序列| | |
|`get_loc(key[,method,tolerance])`|单个位置|`int`、`slice`、`bool`|索引项可重复|
|`get_indexer(target[,method,limit,...])`|多个位置|`int`、`ndarray`|索引项无重复、模糊匹配、无匹配置 `-1`|
|`pd.interval_range`|按频率创建区间索引|`start`、`end`、`freq`| |支持日期类型|

-	`IntervalIndex` 区间索引：围绕 `Interval` 构建的容器
	-	特点
		-	`.loc`、`[]` 可通过标量抽取包含该变量的区间
		-	`.loc`、`[]` 通过 `Interval` 筛选时，则要求完全匹配
	-	说明
		-	`pd.cut`、`pd.qcut` 切分结果 `Categorical.categories` 即为 `IntervalIndex`

###	`MultiIndex`

|`MultiIndex` 类、方法、属性|描述|返回值|说明|
|-----|-----|-----|-----|
|`MultiIndex([levels,codes,sortorder,...])`|按层次构造| |类似 `from_arrays`|
|`IndexSlice[]`|包装索引以方便使用| | |
|`MultiIndex.from_arrays(arrays[,sortorder,...])`|按层次构造| |每项作为索引层（级）|
|`MultiIndex.from_tuples(tuples[,sortorder,...])`|按元素构造| |每项作为索引项|
|`MultiIndex.from_product(iterables[,...])`|按笛卡尔积创建| |各项创建笛卡尔积作为索引项|
|`MultiIndex.from_frame(df[,sortorder,names])`|从 `DF` 创建| |类似 `from_tuples`|
|`names`|各层级名| | |
|`dtypes`|各层级数据类型| | |
|`nlevels`|层级数|`int`| |
|`levshape`|各层级因子数| |
|`levels`|各层因子|`FrozenList`| |
|`codes`|各层编码|`FronzenList`| |
|`to_frame`|转换为 `DataFrame`| |`pd.DF`| |
|`remove_unused_level`|移除未使用级别| | |
|`is_lexsorted`|索引是否词法排序| |`lexsort_depth` 属性可获取已词法排序层级数|

-	`MultiIndex` 是
	-	可视为元组数组，其中每个元组唯一
		-	元组本身可以作为索引值，需与 `MultiIndex` 使用元组抽取区分
		-	抽取数据时也通过元组方式指定多层级索引，需与多轴向索引区分

-	说明事项
	-	创建 `DF` 时，可直接传递嵌套列表指定索引，构造方式同 `from_arrays`
	-	多层级索引可仅指定部分层级抽取数据
		-	可以以元组指定多层级，形式、行为类似多轴向抽取，但优先级更高
		-	切片抽取未 *lexsorted* 多层级索引将 `raise UnsortedIndexError`
		-	非切片抽取未 *lexsorted* 多层级索引时会 `raise PerformanceWarning`
	-	`MultiIndex` 保存有所有被定义的索引层级，即使未被使用
		-	避免重复计算级别，提高性能
	-	可设置选项 `display.multi_sparse`，控制高层级索引是否稀疏化展示

####	*MultiIndex* 变换

|`MultiIndex` 类、方法、属性|描述|返回值|说明|
|-----|-----|-----|-----|
|`set_names(names[,level,inplac])`|命名索引|`S`| |
|`set_levels(levels[,level,...])`|设置各层因子|`S`|全等映射|
|`set_codes(codes[,level,...])`|设置各层编码序列|`S`|修改各层组合关系|
|`to_flat_index()`|转换位元组索引|`C=1`| |
|`to_frame([index,name])`|转换为 `DF`| | |
|`sort_level([level,ascending,...])`|排序|`S`| |
|`droplevel([level])`|丢弃层级|`C-`| |
|`swaplevel([i,j])`|交换层级|`S`| |
|`reorder_levels(order)`|指定排序|`S`| |
|`remove_unused_levels()`|移除未使用层级|`C-`| |
|`get_loc(key[,method])`|单标签（多层级）位置|`int`、`slice`、`bool`|索引项可重复|
|`get_locs(seq)`|标签序列位置|`ndarray`| |
|`get_loc_level(key[,level,...])`|类似 `get_loc`，允许指定层级|`(loc,REMAIN_LEVEL_INDEX)`| |
|`get_indexer(target[,method,limit,...])`|多个标签位置|`int`、`ndarray`|索引项无重复、模糊匹配、无匹配置 `-1`|
|`get_level_values(level)`|层级值|`C=1`| |

###	*DatetimeIndex*

|`DatetimeIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`DatatimeIndex([data,freq,tz,normalize,...])`|不可变 `datatime64` 数据| | |
|`year`、`month`、`day`|年、月、日| | |
|`hour`、`miniute`、`second`|时、分、秒| | |
|`microsecond`、`nanosecond`|微秒、毫秒| | |
|`date`|`datetime.date` 序列|`ndarray(datetime.date)`| | |
|`time`|`datetime.time` 序列|`ndarray(datetime.time)`| | |
|`timez`|带时区 `datetime.time` 序列|`ndarray(datetime.time)`| | |
|`dayofyear`、`day_of_year`|年中日| | |
|`dayofweek`、`day_of_week`、`weekday`|周中日| |周一为 0|
|`quarter`|季度| | |
|`tz`|时区| | |
|`freq`|`freq` 对象| |未设置则 `None`|
|`freqstr`|`freq` 对象字符串| |同上|
|`inferred_freq`|推测的频率| | |
|`is_month_start`、`is_month_end`|月初、月末| | |
|`is_quarter_start`、`is_quarter_end`|季度初、季度j末| | |
|`is_year_start`、`is_year_end`|年初、年末| | |
|`is_month_start`、`is_month_end`|月初、月末| | |
|`is_month_start`、`is_month_end`|月初、月末| | |
|`is_leap_year`|闰年| | |

-	`DatetimeIndex` 说明
	-	继承有有 `Int64Index`，支持一定算数运算
	-	时间均以本地时计，转换时区时时间会相应改变

> - *DatetimeIndex API*：<https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html#datetimeindex>
> - `DatetimeIndex`：<https://www.pypandas.cn/docs/user_guide/timeseries.html#timeseries-overview>

####	筛选

|`DatetimeIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`indexer_at_time(time[,asof])`|准确时间位置|`ndarray`|不考虑日期|
|`indexer_between_time(start_time,end_time,...)`|时间区间位置|`ndarray`|不可包含日期|

####	修改

|`DatetimeIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`normalize()`|时间置为午夜|`S`|用于时间可忽略场合|
|`strftime(date_format)`|转换为字符串索引|`Index`| |
|`snap([freq])`|偏移至最近 `freq`|`S`|仅支持年、月|
|`round(freq[,ambiguous,...])`|舍入至最近 `freq`|`S`|不支持年、月|
|`floor(freq[,ambiguous,...])`|下舍至最近 `freq`|`S`|同上|
|`ceil(freq[,ambiguous,...])`|上舍至最近 `freq`|`S`|同上|
|`tz_localize(tz[,ambiguous,...])`|添加时区信息|`S`| |
|`tz_convert(tz)`|转换时区|`S`| |
|`month_name([locale])`|本地化月名称| | |
|`day_name([locale])`|本地化日名称| | |
|`shift([periods,freq])`|偏移索引项值|`S`|时序索引兼容方法|

####	类型转换

|`DatetimeIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`to_periods(freq)`|转换为 `PeriodIndex`| | |
|`to_periodsdelta(freq)`| | | |
|`to_pydatetime()`|转换为 `datetime.datetime`|`ndarray`| |
|`to_series([keep_tz,index,name])`|转换为 `Series`| |索引、值相同|
|`to_frame([index,name])`|转换为 `DF`|`C=1`| |

####	值聚集

|`DatetimeIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`mean([skipna,axis])`|均值| | |
|`std([axis,ddof,skipna])`|标准差| | |

###	TimedeltaIndex

|`TimedeltaIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`TimedeltaIndex([data,unit,freq,closed,...])`|不可变 `timedelta64` 序列| | |
|`days`、`seconds`、`microseconds`、`nanoseconds`|日、秒| | |
|`components`|`DF` 方式返回日、时、分、秒等成分|`DF`| |
|`inferred_freq`|猜测的 `freq`| | |
|`to_pytimedelta()`|转换为 `datetime.timedelta`|`ndarray`| |
|`to_series([keep_tz,index,name])`|转换为 `Series`| |索引、值相同|
|`to_frame([index,name])`|转换为 `DF`|`C=1`| |
|`round(freq[,ambiguous,...])`|舍入至最近 `freq`|`S`| |
|`floor(freq[,ambiguous,...])`|下舍至最近 `freq`|`S`| |
|`ceil(freq[,ambiguous,...])`|上舍至最近 `freq`|`S`| |
|`mean([skipna,axis])`|均值| | |

> - *TimedeltaIndex API*：<https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html#timedeltaindex>

###	PeriodIndex

|`PeriodIndex` 类、属性、方法|描述|返回值|说明|
|-----|-----|-----|-----|
|`PeriodIndex([data,ordinal,freq,closed,...])`|表示时期的不可变有序序列| | |
|`year`、`month`、`day`|年、月、日| | |
|`hour`、`miniute`、`second`|时、分、秒| | |
|`quarter`|季度| | |
|`week`、`weekofyear`|周序| | |
|`dayofyear`、`day_of_year`|年中日| | |
|`daysinmonth`、`days_in_month`|月中日| | |
|`dayofweek`、`day_of_week`、`weekday`|周中日| |周一为 0|
|`freq`|`freq` 对象| |未设置则 `None`|
|`freqstr`|`freq` 对象字符串| |同上|
|`is_leap_year`|闰年| | |
|`start_time`|时期起始点|`DatetimeIndex S`| |
|`end_time`|时期结束点|`DatetimeIndex S`| |
|`asfreq([freq,how])`|转换为指定 `freq`| |用时期首、尾替代|
|`strftime(date_format)`|转换为字符串索引|`Index`| |
|`to_timestamp([freq,how])`|转换为 `DatetimeIndex`|`S`|用时期首、尾替代|

-	`PeriodIndex` 说明
	-	继承有有 `Int64Index`，支持一定算数运算
	-	时间均以本地时计，转换时区时时间会相应改变

> - *PeriodIndex API*：<https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html#periodindex>

##	`Series` 数据类型

|数据|数据类型 *Dtype*|标量|数组|字符串别名|
|-----|-----|-----|-----|-----|
|*tz-aware datetime*|`DatetimeTZDtype`|`Timestamp`|`arrays.DatetimeArray`|`datetime64[ns,<TZ>]`|
|*Categorical*|`CategoricalDtype`| |`Categorical`|`category`|
|*period(time spans)*|`PeriodDtype`|`Period`|`arrays.Peroidarray`|`period[<FREQ>]`|
|*timedelta*| |`Timedelta`|`TimedeltaArray`|`timedelta64[ns]`|
|*sparse*|`SparseDtype`| |`arrays.SparseArray`|`Sparse`、`Sparse[int]`、`Sparse[float]`|
|*intervals*|`IntervalDtype`|`Interval`|`arrays.IntervalsArray`|`interval`、`Interval`、`Interval[<NPDT>]`、`Interval[datetime64[ns,<TZ>]`、`Interval[timedelta64[<FREQ>]]`|
|*nullable integer*|`Int64Dtype`、`Int32Dtype`、`Int16Dtype`、`Int8Dtype`| |`arrays.IntegerArray`|`Int64`、`UInt64` 等|
|*Strings*|`StringDtype`|`str`|`arrays.StringArray`|`String`|
|*Boolean*|`BooleanDtype`|`bool`|`arrays.BooleanArray`|`boolean`|

-	数据类型说明
	-	由 *Numpy* 提供以下数据类型的支持
		-	`float`
		-	`int`
		-	`bool`
		-	`timedelta64[ns]`
		-	`datetime64[ns]`：不支持时区
	-	需数据类型作 `dtype` 参数时，可用j
		-	数据类型类实例
		-	字符串别名

> - 数据类型：<https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#dtypes>

###	*Categorical Data*

|`.cat` 访问器|描述|返回值|说明|
|-----|-----|-----|-----|
|`categories`|因子|`Index`| |
|`ordered`|是否有序| | |
|`as_ordered(inplace)`|设为有序| | |
|`as_unordered(inplace)`|设为无序| | |
|`rename_categories(new_categories[,inplace])`|重命名因子|`S`|不影响序列值|
|`add_categories(new_categories[,inplace])`|添加因子|`S`| |
|`remove_categories(new_categories[,inplace...])`|移除因子|`S`|被移除值以 `np.nan` 替代|
|`remove_unused_categories([inplace])`|移除未出现因子|`S`| |
|`set_categories(new_categories,ordered,rename)`|设置因子| | |
|`reorder_categories(new_categories,ordered[,inplace])`|重排序因子|`S`| |
|`Categorical.from_codes(codes,categories,ordered)`|从编码、因子创建分类数组| | |
|`pd.api.types.union_categoricals`|合并分类| | |改变编码|

-	`CategoricalDtype`：分类数据类型（支持有序、无序）
	-	特点
		-	可通过 `catetogries`、`ordered` 完全描述
		-	内部以因子数组、编码数组方式存储，重复数据较多时节省空间
		-	数据只能取类型值、`np.nan`（`pd.NA` 会被自动转换）
		-	对任意实例有 `== "category" == CategoricalDtype(None, False)`
		-	值修改仅应用于因子而不是逐元素，重复值较多时效率高
	-	方法说明
		-	若为无序分类，则 `min`、`max` 将 `raise TypeError`
		-	算数操作、基于算数操作方法将 `raise TyperError`
		-	`value_counts` 等聚集方法中，未出现因子也将在结果中列出
	-	分类数据类型保持、转换说明
		-	切片、列表获取将返回分类数据，标量获取将返回因子值
		-	赋值要求右值为因子值、或同类型分类数据
		-	数据拼接、联结时，仅在行（列）数据类型相同才保持分类类型，否则转换为因子值类型
		-	`.dt`、`.str` 访问器在 `.cat.categories` 为相应类型时可用，返回结果不保持分类类型

-	`.cat` 访问器：访问数据值分类属性的对象
	-	`pd.core.arrays.categorical.CategoricalAccessor` 别名
	-	可借此调用向量化分类类型相关方法
		-	同 `CategoricalIndex` 方法，但均包含 `inplace` 参数

> - *Categorical* 数据：<https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>
> - *CategoricalDtype API*：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.CategoricalDtype.html>
> - `.cat` 访问器 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.categories.html>
> - `.cat` 访问器：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cat.html>

###	*Text Data*

|`.str` 访问器（部分）|描述|返回值|说明|
|-----|-----|-----|-----|
|`split([pat,n,expand,...])`|切分|`C+`|置位 `expand` 将返回 `DF`|
|`rsplit([pat,n,expand,...])`|右侧开始切分|同上|
|`cat([others,sep,na_rep,...])`|拼接、连接|`C-`|支持自聚集、键连接|
|`get(i)`、`[i]`|抽取元素| |值为列表、字典等时同样支持此方法|
|`extract(pat[,flags,expand])`|模式抽取首个|`C+`| |支持正则|
|`extractall(pat[,flags])`|模式抽取全部|`C+`| |支持正则|
|`replace(pat,repl[,n,...])`|替换|`S`|支持正则|
|`contains(pat[,case,flags,...])`|包含|`bool S`|支持正则|
|`match(pat[,case,flags,...])`|匹配|`bool S`|支持正则|
|`get_dummies([sep])`|根据字符串创建指示器|`C+`|`sep` 表示或|

-	`StringDtype`：新式文本数据类型
	-	相较 `object` 类型存储文本数据，类型的确定带来优势

-	`.str` 访问器：访问数据值文本属性（字符串、字符串列表）的对象
	-	`.str` 访问器处理逻辑、方法返回值有优化
		-	本质上是对各项向量化调用方法，则仅需值支持方法即可，不必须为字符串
	-	可借此调用向量化字符串相关方法
		-	`str` 同名方法（细节有差异）：`split`、`lower`、`replace` 等
		-	正则相关：`replace`、`extract`、`contains`
		-	创建指示器：`get_dummies`
		-	其他：`get`、`removeprefix`、`cat`

> - 文本处理：<https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html>
> - `str` 访问器 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.capitalize.html>
> - `str` 访问器：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.html>

###	时序数据

-	`DatetimeTZDtype` 数据类型：时点
	-	即对应标量 `pd.Timestamp` 类型
		-	*Pandas* 中替代 `datetime.datetime` 的类型
		-	继承有 `Int64Index`，支持一定算数运算
	-	支持指定单位（缺省 `ns`）、时区

-	`timedelta64[<UNIT>]` 数据类型：时间差，无对应数据类型
	-	即对应标量 `pd.Timedelta` 类型
		-	*Pandas* 中替代 `datetime.timedelta` 的类型
		-	继承有 `Int64Index`，支持一定算数运算
	-	支持指定单位（缺省 `ns`）
		-	但非 `ns` 为单位时，即等价于 `np.int64` 类型

-	`PeriodDtype` 数据类型：一段时期
	-	即对应标量 `pd.Period` 类型
		-	继承有 `Int64Index`，支持一定算数运算

-	`.dt` 访问器：访问数据 *datetimelike* 属性的对象
	-	`pd.core.index.accessors.CombineDatetimelikeProperties` 别名
	-	可借此访问向量化时序类型相关方法、属性
		-	方法同 `DatetimeIndex` 方法

> - `Timestamp` 类型：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html>
> - `Timedelta` 类型：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html>
> - `Period` 类型：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Period.html>
> - `.dt` 访问器：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html>
> - `.dt` 访问器 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.date.html>

###	*Interval*

-	`IntervalDtype` 数据类型：不可变的、类似切片的区间
	-	即对应标量 `pd.Interval` 类型
		-	鸭子类型，仅存储区间端点类型、开闭类型

> - `Interval` 类型：<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Interval.html>

##	*Pandas* 设置

|函数|描述|入参|返回值|说明|
|-----|-----|-----|-----|-----|
|`get_option`|获取选项|`pat`| |`pat` 支持正则，下同|
|`set_option`|设置选项|`pat`、`value`| | |
|`reset_option`|重置选项为默认值|`pat`| | |
|`describe_option`|打印选项|`pat`| | |
|`option_context`|选项设置上下文管理器|`*args`（连续的键值）| | |

-	选项设置 *API* 说明
	-	选项名为 `.` 分隔、大小写敏感的格式

> - 选项设置：<https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html>
> - 选项设置 *API*：<https://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html>

##	*I/O* 工具

|数据格式|数据描述|*Reader*|Writer*|
|-----|-----|-----|-----|
|文本|`csv`|`read_csv`|`to_csv`|
|文本|等宽文本|`read_fwf`| |
|文本|`json`|`read_json`|`to_json`|
|文本|`html`|`read_html`|`to_html`|
|文本|`LaTeX`| |`Styler.to_latex`|
|文本|`xml`|`read_xml`|`to_xml`|
|文本|粘贴板|`read_clipboard`|`to_clipboard`|
|二进制|MS Excel|`read_excel`|`to_excel`|
|二进制|OpenDocument|`read_excel`| |
|二进制|HDF5|`read_hdf`|`to_hdf`|
|二进制|Feather|`read_feather`|`to_feather`|
|二进制|Parquet|`read_parquet`|`to_parquet`|
|二进制|Msgpack|`read_msgpack`|`to_msgpack`|
|二进制|Stata|`read_stata`|`to_stata`|
|二进制|SAS|`read_sas`| |
|二进制|SPSS|`read_spss`| |
|二进制|Pickle|`read_pickle`|`to_pickle`|
|SQL|`sql`|`read_sql`|`to_sql`|
|SQL|Google Big Query|`read_gbq`|`to_gbq`|
|二进制|`xarray`| |`to_xarray`|

> - *I/O* 工具：<https://www.pypandas.cn/docs/user_guide/io.html>
> - *I/O* 工具：<https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>
> - *I/O API*：<https://pandas.pydata.org/pandas-docs/stable/reference/io.html>

###	*CSV I/O* 参数

|`read_csv` 参数|含义|实参类型|说明|
|-----|-----|-----|-----|
|`filepath_or_buffer`|数据来源|文件、*I/O* 对象| |
|`sep`|分隔符|`,`、`str`| |
|`delimiter`|`sep` 别名|`None`、`str`| |
|`delim_whitespace`|使用空白坐分隔符|`False`|等价 `sep="\s+"`|
|`header`|列名行|`None`、`int`、`list[int]`|其中未指定的间隔行被省略，其他参数指定跳过行不计入计数|
|`names`|列名|`list`|不应与 `header` 同时传递有效值|
|`index_col`|行名行|`None`、`int`、`str`、`False`、`list`|行名列|
|`usecols`|有效列|`None`、`list`、`callable`|`callable` 接受列名（处理后），返回布尔值|
|`sequeeze`|单列转换为 `pd.Series`|`False`| |
|`mangle_dupe_cols`|保留重名列|`True`|重名列添加后缀，否则不断替换数据|
|`prefix`|行名前缀|`None`、`str`| |
|`dtype`|数据类型|*dtype*、`dict`| |
|`converters`|数据值转换|`{K: callable}`| |
|`true_values`|真值列表|`list`| |
|`false_values`|假值列表|`list`| |
|`skipinitialspace`|跳过分隔符后空白符|`False`| |
|`skiprows`|跳过头行|`int`、`list[int]`| |
|`skipfooter`|跳过尾行|`int`| |
|`nrows`|读取行数|`int`| |
|`low_memory`|内部分块处理文件|`True`|可能导致同列类型不一致|
|`memory_map`|直接从内存映射|`False`|要求数据以缓存方式传入|
|`na_value`|`NA` 值列表|`None`、`str`、`list`、`dict`| |
|`keep_default_na`|是否包含默认 `NA` 值列表|`True`| |
|`na_filter`|检查 `NA`|`True`|置 `False` 可提高效率|
|`verbose`|指出 `NA` 位于非数值列|`False`| |
|`skip_blank_lines`|跳过空行|`True`| |
|`parse_dates`|解析日期|`False`、`list`、`list[list]`、`dict`|支持多列合并解析|
|`inter_datetime_format`|推测日期格式|`False`| |
|`keep_date_col`|多列合并解析时保留原列|`False`| |
|`date_parser`|日期解析器|`callable`|缺省为 `dateutil.parser.parser`|
|`dayfirst`|日在月前格式|`False`| |
|`cache_dates`|缓存加速|`True`| |
|`iterator`|返回 `TextFileReader` 对象用于迭代、`get_chunk()`|`False`| |
|`chunksize`|返回 `TextFileReader` 对象用于迭代|`None`、`int`| |
|`compression`|数据压缩方式|`infer`、`str`、`dict`|`dict` 用于指定压缩详细信息|
|`thousands`|千分隔符|`None`、`str`| |
|`decimal`|用于确定实数的分隔符|`.`、`str`| |
|`float_precision`|浮点精度|`None`、`high`、`round_trip`| |
|`lineterminator`|换行符|`None`、`str`| |
|`quotechar`|括起引用所用的字符|`str`| |
|`quoting`|引用行为控制|`0`、`int`|应为模块常数值 `csv.QOUTE_`|
|`doublequote`|是否将连续的引用字符视为单个|`True`| |
|`escapechar`|`csv.QUOTE_NONE` 时转义分隔符的字符|`None`| |
|`comment`|注释引导字符|`None`| |
|`encoding`|编码|`None`、`str`| |
|`dialect`|格式风格|`None`、`str`、`csv.Dialect`| |
|`on_bad_lines`|处理异常行|`error`、`warn`、`skip`| |

###	*Excel*、*OpenDocument* 参数

-	`read_excel`、`write_excel` 参数、语义类似 *CSV* 文件 *API*
	-	`engine`：文件解析引擎，不同格式依赖不同模块
		-	`.xlsx` 依赖 `openpyxl`、`xlsxwriter`
		-	`.xls` 依赖 `xlrd`、`xlwt`
		-	`.xlsb` 依赖 `pyxlsb`
		-	`.odf` 依赖 `odfpy`
	-	`sheet_name`：读取子表名称

