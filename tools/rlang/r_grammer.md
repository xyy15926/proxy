---
title: R语法
tags:
  - RLang
categories:
  - R语言
date: 2019-03-21 17:27:15
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: R语法
---

##	生存

###	配置

####	注释

-	R语言的注释和其他脚本语言类似，使用`#`注释行，但是R没有
	多行注释语法
-	可以使用`if(FALSE)`语句进行“注释”
	```r
	if(FALSE) {
		"comments"
	}
	```

####	工作路径

```r
getwd()
	# 返回当前工作路径，默认`~/Documents`
setwd(/path/to/dir)
	# 设置当前工作路径
```

###	变量

-	R中有效的变量名称由字母、数字、点、下划线组成，变量名以
	字母、后不跟数字的点开头
-	R中变量为动态类型，可多次更改变量数据类型
-	变量无法被声明，在首次赋值时生成

####	赋值

```r
var.1 = c(0, 1, 2, 3)
	# 不常用
var.2 <- c("learn", "R")
	# 常用方法
var .3 <<- c(3, 1, TRUE, 2+3i)
	# 特殊赋值符，可以扩展变作用域为“整个”工作空间
	
c(TRUE, 1) -> var.4
	# 右赋值符
c(TRUE, 2+3i) ->> var.5
```

####	搜索

`ls`函数可以搜索当前工作空间中所有可用变量

```r
ls(
	pattern = "var_pattern",
	all.name = FALSE/TRUE
)
	# `pattern`：使用模式匹配变量名
	# `all.name`：开头变量默认隐藏，可通过设置参数展示
```

####	删除

`rm`函数可以删除变量

```r
rm(var.3)

rm(list = ls())
	# 删除所有变量
```

###	数据导入

####	`edit`

```r
mydata <- data.frame(
	age = numeric(0),
	gender = character(0),
	weight = numeric(0)
)
	# `numeric(0)`类似的赋值语句创建指定模式但不含数据变量
mydata <- edit(mydata)
	# `edit`会调用允许手动输入数据的文本编辑其
	# `edit`在副本上操作，需要重新给变量
fix(mydata)
	# `mydata <- edit(mydata)`等价写法，直接在原对象上操作
```

####	`read.table`

```r
DF <- read.table(
	file(/path/to/file),
	header = TRUE/FALSE,
	sep = " \t\n\r"/",",
	row.names = c(),
	col.names = c("V1", "V2", ...)/c(str),
	na.strings = NULL/c(str)),
	colClasses = NULL/c("numeric", "character", "NULL"),
	quote = "'""/str,
	skip = 0/int,
	stringAsFactors = TRUE/FALSE,
	test = str
)
```
-	说明：从**带分隔符**的文本文件中导入数据

-	参数
	-	`header`：第一行是否包含变量名
	-	`sep`：分隔符，默认数个空格、tab、回车、换行
	-	`row.names`：指定行标记符
	-	`col.names`：指定DF对象列名
	-	`na.strings`：表示缺失值的字符串向量，其包含字符串
		读取时转为NA
	-	`colClasses`：设置DF对象每列数据模式
		-	"NULL"表示跳过
		-	长度小于列数时开始循环
		-	读取大型文本可以提高速度
	-	`quote`：字符串划定界限，默认`"'`
	-	`StringAsFactor`：标记字符向量是否转换为factor
		-	`colClasses`优先级更高
		-	设为FALSE可以提升读取速度
	-	`text`：读取、处理的字符串，而不是`file`

###	常用函数

```r
print()
	# 浏览对象取值
str()
	# 查看对象结构
ls()
	# 管理对象
remove()
rm()
	# 删除指定对象
```

##	数据模式（类型）

-	数据模式是指R存储数据的方式
-	即从存储角度对R数据对象划分
-	`class`函数就是返回数据模式（类型）

###	Logical

只需要1byte存储

-	`TRUE/T`
-	`FALSE/F`

###	Integer

占用2-4byte

-	`2L`、`0L`

###	Numeric

可进一步分

-	`float`占用4byte
-	`double`占用8byte

R中数值型数据默认为`double`

-	`12.3`、`4`

###	Complex

-	`comlplex`：`3+2i`

###	Character

R中`'`、`"`对中的任何值视为字符串

-	`'`、`"`必须在开头、结尾成对存在
-	`'`、`"`结尾的字符串中，只能插入对方

####	`paste`

连接多个字符串

```r
Chars = paste(
	...,
		# 要组合的任意数量变量
	sep = " ",
		# 分隔符
	collapse = NULL)
		# 消除两个字符串之间空格
```

####	`format`

将数字、字符串格式为特定样式

```r
Chars = format(
	x([num, chars],
		# 向量
	digits(int),
		# 显示的总位数
	nsmall(int),
		# 小数点右边最小位数
	scientific=FALSE/TRUE,
		# `TRUE`则显示科学计数法
	width(int),
		# 在开始处填充空白来显示的最小宽度
	justify = c("left", "right", "centre", "none"))
		# 字符串显示位置
```

####	`nchar`

计算包括空格在内的字符串长度

```r
int = nchar(
	x(chars)
)
```

####	`toupper`、`tolower`

改变字符串大小写

```r
chars = toupper(
	x(chars)
)
chars = tolower(
	x(chars)
)
```

####	`substring`

获取字符串子串

```r
chars = substring(
	x(chars),
	first(int),
	last(int)
)
	# 包括头尾
```

-	R对象是指可以赋值给变量的任何事物，包括常量、数据结构、
	函数
-	对象都拥有某种模式，描述对象如何存储
-	对象拥有某个“类”，向`print`这样的泛型函数表明如何处理
	此对象

###	Raw

-	`raw`：`v <- charToRaw("Hello")`（byte类型）

##	结构角度划分R对象

###	Vector

用于存储数值型、字符型、逻辑型数据的一维**数组**

-	单个向量中的出数据必须拥有相同的类型、模式（数值型、字符
	型、逻辑型）
-	R中没有标量，标量以单元素向量形式出现

```r
apple <- c("red", "green", "yellow") 
apple[1]
	# 访问单个元素，从1开始
apple[1: 3]
	# 切片，闭区间
apple[7] = "seven"
	# 将值赋给某个向量、矩阵、数组或列表中一个不存在的元素时
		# R将自动扩展其以容纳新值，中间部分设为`NA`

is.vector(apple)
```

####	创建向量

```r
rep(start: end, each=repeat_time)
	# 元素重复
rep(start: end, times=repeat_time)
	# 向量重复

seq(from=start, to=end, by=step)
	# 指定步长
seq(from=start, to=end, length=len)
	# 指定个数

vector(length=len)
	# 元素为`FALSE`
```

####	访问向量元素

```r
a[1]
a[1:2]
a[c(1,3)]
a[c(T, F, T)]

a[-c(1:2)]
	# 负号不能用于逻辑向量
```

###	Matrix

二维数组：组织具有相同存储类型的一组变量

-	每个元素都拥有相同的模式（数值型、字符型、逻辑型）

####	创建矩阵

```r
mtx <- matrix(
	vector,
	nrow(int)
	ncol(int),
	byrow = FALSE/TRUE,
	dimnames = list(
		c(row_names),
		c(col_names)
	)
)

is.matrix()

cbind()
	# 将多个已有向量（列）合并为矩阵
```

####	矩阵信息

```r
dim(mtx)
	# 显示矩阵行、列
colnames(mtx)
colnames(mtx[, col_start: col_end])
rownames(mtx)
ronnames(mtx[row_start: row_end, ])
```

####	访问矩阵元素

```r


```

###	Array

类似于矩阵，但是维度可以大于2

-	其中元素也只能拥有一种模式

```r
arr <- array(
	vector,
	dimensions(c(int)),
	dimnames = c(dim_names)
)
```

###	Data.Frame

数据帧是表、二维数组类似结构

-	不同的列可以包含不同的模式
-	每列包含一个变量的值，每行包含来自每列的一组值
-	数据帧的数据可是数字、因子、字符串类型

```r
df <- data.frame(
	col1(c()),
	col2(c()),
	...,
	row.names = coln
)
	# `row.names`：指定实例标识符，即index

emp.data <- data.frame(
	emp_id = c(1:5),
	emp_name = c("Rick","Dan","Michelle","Ryan","Gary"),
	salary = c(623.3,515.2,611.0,729.0,843.25),
	start_date = as.Date(c("2017-01-01", "2017-09-23",
		"2017-11-15", "2017-05-11", "2018-03-27")),
	stringsAsFactors = FALSE
)
	# 创建DF
```
####	统计性质

```r
str(emp.data)
	# 可以获得DF结构

summary(emp.data)
	# 获得DF的统计摘要、性质
```

####	筛、删、减

```
emp.data.cols <- data.frame(
	emp.data$emp_name,
	emp.data$salary)
	# 列名称获取DF中特定列

emp.data.rows <- emp.data([1:2, ]
	# 行切片获取特定行

emp.data.rows_2 <- emp.data[c(3, 5), c(2, 4)]
	# 行、列list获取特定行、列

emp.data$dept <- c("IT", "Operations", "IT", "HR", "Finance")
	# 新列名添加新列

emp.newdata <- data.frame(
	emp_id = c(6: 8),
	emp_name = c("Rasmi","Pranab","Tusar"),
	salary = c(578.0,722.5,632.8), 
	start_date = as.Date(c("2013-05-21","2013-07-30","2014-06-17")),
	dept = c("IT","Operations","Fianance"),
	stringsAsFactors = FALSE
)
emp.finaldata <- rbind(emp.data, emp.newdata)
	# `rbind`将新DF同原DFconcat，达到添加新行
```
####	拼、接

#####	`rbind`

结合两个DF对象行

```r
city <- c("Tampa", "Seattle", "Hartford", "Denver")
state <- c("FL", "WA", "CT", "CO")
zipcode <- c(33602, 98104, 06161, 80294)

address <- cbind(city, state, zipcode)
	# `cbind`连接多个向量创建DF

new.address <- data.frame(
	city = c("Lowry", "Charlotte"),
	state = c("CO", "FL"),
	zipcode = C("80230", "33949"),
	stringAsFactors = FALSE
)
	# 使用`data.fram`创建DF

all.address <- rbind(
	address,
	new.address
)
	# `rbind`结合两个DF的行
```

#####	`merge`

根据两DF列进行merge

```r
lirary(MASS)
	# 加载数据集

merged.Pima <- merge(
	x = Pima.te,
	y = Pima.tr,
	by.x = c("bp", "bmi"),
	by.y = c("bp", "bmi")
)
	# 根据DF某（些）列merge
```

#####	`melt`、`cast`

```r
library(MASS)
library(reshape2)
melton.ships <- melt(
	ships,
	id = c("type", "year")
)
	# `melt`将剩余列转换为`variable`、`value`标识

recasted.ship <- cast(
	molten.ship,
	type+year~variable,sum
)
	# 和`melt`相反，以某些列为“轴”合并
```

####	绑定

#####	`attach`、`detach`

```
attach(emp.data)
	# `attach`可以将数据框加入R的搜索路径
	# 之后R遇到变量名之后，将检查搜索路径的数据框
	# 注意，如果之前环境中已经有df对象列同名变量，那么原始
		# 对象优先，不会被覆盖

	emp.data.cols.copy <- data.frame(
		emp_name,
		salary
	)
		# 否则需要之前一样使用`$`
detach(emp.data)
	# 将数据框从搜索路径移除
```

#####	`with`

```python
with(emp.data, {
	emp.data.cols.copy <<- data.frame(
		emp_name,
		salary
	)
})
	# `{}`中的语句都针对`emp.data`执行，如果只有一条语句，
		# 花括号可以省略
	# `with`语句内`<-`赋值仅在其作用于内生效，需要使用`<<-`
		# 特殊赋值符保存至“全局”变量中
```

###	Factor

分类变量、有序变量在R中称为因子，其决定了数据的分析方式、
如何进行视觉呈现

```python
fctr <- factor(
	vector(c(factors)),
	ordered = FALSE/TRUE,
	levels = c(ordered_unique_factors),
	labels = c(factor_labels)

	# `ordered`：默认不是有序变量
	# `levels`：指定factors的“排序”，确定映射的整数值，
		# 对于分类变量也可以设置
		# 没有在`levels`中显式指定的factor视为缺失
	# `labels`：设置各factor labels，输出则按照labels输出
		# 注意`labels`顺序必须和`levels`一致
		# 对数值型factor尤其有用
)
```
-	`factor`以整形向量的形式存储类别值
	-	整数取值范围为`1~k`，`k`为定性（分类、有序）变量中
		唯一值个数
-	同时一个由字符串（原始值）组成的内部向量将映射到这些整数
	-	分类变量：字符串映射的整数值由字母序决定
	-	有序变量：按字母序映射可能与逻辑顺序不一致，可以使用
		参数`levels`指定顺序

###	List

一些对象、成分的有序集合

-	允许整合若干（可能无关）的对象到单个对象下
-	很多R函数结果运行结果以列表形式返回，由调用者决定使用
	其中何种成分

```r

l <- list(
	[name1 =]object1,
	[name2 =]object2,
	...
)
	# 可以给列表中的对象命名
	# 命名成分`l$name1`也可以正常运行


list1 <- list(c(2, 5, 3), 21, 3, sin)
print(list1)
print(class(list1))
```

##	运算符

###	算术运算符

算术操作符作用与向量的每个元素

```r
v <- c(2, 5.5, 6)
t <- c(8, 3, 4)
print(v + t)
	# 算术操作符作用与向量的每个元素
print(v - t)
print(v * t)
print(v/t)
print(v %% t)
	# 向量求余
print(v %/% t)
	# 求商
print(v ^ t)
	# 指数运算
```

###	关系运算符

比较两个向量的相应元素，返回布尔值向量

```r
v <- c(2, 5.5, 6, 9)
t <- c(8, 2.5, 14, 9)
print (v > t)
	# 比较两个向量的相应元素，返回布尔值向量
print (v < t)
print (v == t)
print (v != t)
print (v <= t)
print (v >= t)
```

###	逻辑运算符

只适用于逻辑、数字、复杂类型向量，所有大于1的数字被认为是
逻辑值`TRUE`

```r
v <- c(3, 1, TRUE, 2+3i)
t <- c(4, 1, False, 2+3i)
print(v & t)
	# 比较两个向量相应元素，返回布尔值向量
print(v | t)
print(!v)

print(v && t)
	# 只考虑向量的第一个元素，输出单个bool值元素向量
print(v || t)
```

###	其他运算符

```r
t <- 2: 8
	# 为向量按顺序创建一系列数字

v1 <- 8
v2 <- 12
print (v1 %in% t)
print (v2 %in% t)
	# 标识元素是否属于向量

M = matrix(c(2, 6, 5, 1, 10, 4),
	nrow = 2,
	ncol = 3,
	byrow = TRUE)
t = M %*% t(M)
	# 矩阵相乘
```

##	R语句

###	条件

```r
if
else
switch
```

###	循环

```r
repeat
while
for
break
next
```

##	函数

```r
func_name <- function(
	arg_1,
	arg_2,...){

}
```

###	R内置函数

```r
print(seq(32, 44))
print(mean(25: 82))
print(sum(41: 68))
```

###	自定义参数

```r
new.function_1 <- fucntion(){
	# 无参数函数
	for(i in 1: 5){
		print(i ^ 2)
	}
}
new.function_1()

new.function_2 <- function(a, b, c){
	# 有参数函数
	result <- a * b + c
	print(result)
}
new.function_2(5, 3, 11)
	# 按参数顺序调用函数
new.function_2(
	a = 11,
	b = 5,
	c = 3)
	# 按参数名称调用

new.function_3 <- function(a=3, b=6){
	# 含有默认参数函数
	result <- a * b
	print(result)
}
new.function_3
	# 无参数（使用默认参数）
new.function_3(9, 5)
```

###	函数功能延迟计算

```r
new.function <- function(a, b){
	print(a ^ 2)
	print(a)
	print(b)
}
new.function(6)
	# 调用函数能部分执行成功，直到`print(b)`
```

##	包

R语言包是R函数、编译代码、样本数据的集合

-	存储在R环境中名为`library`的目录下
-	默认情况下，只有R安装时提供默认包可用，已安装的其他包
	必须显式加载

```r
.libPaths()
	# 获取包含R包的库位置

library()
	# 获取已安装所有软件包列表

search()
	# 获取当前R环境中加载的所有包
```

###	安装包

-	直接从CRAN安装
	```r
	install.package("pkg_name")
	```

-	手动安装包：从<https://cran.r-project.org/web/packages/available_packages_by_name.html>
	中下载包，将zip文件保存
	```r
	install.package(/path/to/pkg.zip, repos=NULL, type="source")
	```

###	加载包

```r
library("pkg_name",
	lib.loc=/path/to/library)
	# `lib.loc`参数默认应该就是`.libPaths`，一般不用设置
```









