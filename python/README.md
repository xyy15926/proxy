#	Python笔记约定

##	函数书写声明

```python
return = func(essential(type), optional=defaults/type,
	*args, **kwargs)
```

###	格式说明

####	参数

-	`essential`参数：`essential(type)`
	-	没有`=`
	-	在参数列表开头
	-	`(type)`：表示参数可取值类型

-	`optional`参数：`optional=defaults/type`
	-	`defaults`若为具体值，表示默认参数值
		-	默认值首字母大写
	-	默认参数值为`None`
		-	函数内部有默认行为
		-	`None`（默认行为等价参数值）
	-	`type`：之后表示可能取值类型

-	`args`参数：`[参数名]=defaults/type`
	-	首参数值为具体值表示函数默认行为（不是默认值，`args`
		参数没有默认值一说）
	-	其后为可取参数值类型
	-	说明
		-	参数名仅是**标记**作用，不能使用关键字传参
		-	`[]`：代表参数“可选”

-	`kwargs`参数：`[param_name=defaults/type]`
	-	参数默认为可选参数，格式规则类似`args`参数

-	POSITION_ONLY参数：`[param_name](defaults/type)`
	-	POSITION_ONLY参数同样没有默认值一说，只是表示默认
		行为方式（对应参数值）

补充：
-	参数名后有`?`表示参数名待确定
-	参数首字母大写表示唯一参数

####	返回值

返回值类型由返回对象**名称**蕴含

####	对象类型

-		`obj(type)`：`type`表示**包含**元素类型

####	其他

-	DF对象和Series对象都具有的函数属性列出DF对象

##	常用参数说明

以下常用参数如不特殊注明，按照此解释

###	Pandas

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

###	Matplotlib

#	todo

-	`data=dict/pd.DataFrame`

	-	其他
		-	属于kwargs中参数
		-	传参时，相应键值对替代对应参数

###	Numpy

-	`size=None(1)/int/tuple(int)`

	-	含义：ndarray形状
	-	默认：一般`None`，返回一个值

-	`dtype=None/str/np.int/np.float...`

	-	含义：ndarray中数据类型
	-	默认值：`None`，有内部操作，选择合适、不影响精度类型
	-	其他
		-	可以是字符串形式，也可以是`np.`对象形式
-	`order = "C"/"F"/"K"/"A"
	-	含义：NDA对象在内存中的存储方式
		-	"C"：`C`存储方式，行优先
		-	"F"：`Fortran`存储方式，列优先
		-	"K"：原为"C"/"F"方式则保持不变，否则按照较接近
			方式
		-	"A"：除非原为"F"方式，否则为"C"方式
	-	默认值："C"/"K"

>	Numpy包中大部分应该是调用底层包，参数形式不好确认

###	threading

-	`block/blocking = True/False`

	-	含义：是否阻塞
	-	默认：大部分为`True`（阻塞）

-	`timeout = None/num`

	-	含义：延迟时间，单位一般是秒
	-	默认：None，无限时间
