---
title: R可视化
categories:
  - RLang
tags:
  - RLang
  - GGPlot
  - Visualization
date: 2019-03-21 17:27:15
updated: 2021-08-04 19:23:45
toc: true
mathjax: true
comments: true
description: R可视化
---

##	R图形参数

###	使用、保存图形

####	保存图形

```r
png(
	file= "Rplot%03d.png",
	width= 480,
	height= 480,
	units= "px",
		# 图片分辨率
	pointsize= 12,
		# 文字大小
	bg= "white",
	res= NA,
	family= "",
	restoreConsole= TRUE,
	type= c("windows", "cairo"),
	antialias
)
	# `par`语句生效必须放在`png`、`dev.off`中间
	# 这个语句会强制覆盖与`par`语句重合的部分参数，即使这个
		# 函数后调用

pdf(filename)
win.metafile(filename)
jpeg(filename)
bmp(filename)
tiff(filename)
xfig(filename)
postscript(filename)
	# 开启目标图形设备
dev.off()
	# 关闭目标图形设备
	# 绘图语句至于其中，包括`par`

dev.new()
	# 打开新的图形窗口，防止多次绘图覆盖
dev.next()
dev.prev()
dev.set()
	# 选择将图形发送到不同窗口
```

###	`par`

`par`函数设置对整个工作空间图像设置均有效

```r
opar <- par()
	# 无参调用，生成含有当前图形参数设置的列表
opar <- par(no.readonly= TRUE)
	# 生成一个可以修改的当前图形参数列表
par(opar)
	# 恢复为`opar`中的参数设置
par(attr_name)
	# 获取某个属性设置值
```

####	符号、线条

```r
par(
	pch= int,
		# 绘制点时使用的符号
	cex= num,
		# 符号大小，相对于默认大小缩放倍数
	lty= int,
		# 线条类型
	lwd= num,
		# 线条宽度，相对于默认线条宽度的倍数
)
```

####	颜色

```r
par(
	col= str/c(str),
		# 绘图颜色，循环使用
	col.axis= str/c(str),
		# 坐标轴刻度文字颜色
	col.lab= str/c(str),
		# 坐标轴标签颜色
	col.main= str/c(str),
		# 标题颜色
	col.sub= str/c(str)
		# 副标题颜色
	fg= str/c(str),
		# 前景色
	bg= str/c(str),
		# 背景色
)
```

####	文本属性

```r
par(
	cex= num,
		# 文本大小，相对默认
	cex.axis= num,
		# 坐标轴刻度文字的缩放倍数
	cex.lab= num,
		# 坐标轴标签缩放倍数
	cex.main= num,
		# 标题缩放倍数
	cex.sub= num,
		# 副标题缩放倍数
)
```

####	字体、字号、字样

```r
par(
	font= int,
		# 绘图使用字体样式
		# 1：常规、2：粗体、3：斜体、4：粗斜体
	font.axis= int,
		# 坐标轴刻度字体样式
	font.lab= int,
	font.main= int,
	font.sub= int,
	ps= num,
		# 字体磅值
		# 文本最终大小为`cex*ps`
	family= "serif"/"sans"/"mono"/str,
)
```

####	图形、边界尺寸

```r
par(
	pin= c(width, height),
		# 图片尺寸，单位英寸
	mai= c(bot, left, top, right),
		# 边界大小，单位英寸
	mar= c(bot, left, top, right),
		# 边界大小，单位英分（1/12英寸）
		# 默认值`c(5, 4, 4, 2)+ 0.1`
	mgp= c(axis_labels, tick_title, tick),
		# 坐标轴标签、刻度标签、刻度线位置
		# 图形边缘为0，单位为行
)
```

###	文本、自定坐标、图例

####	`title`

`title`函数可以为图形添加标题、坐标轴标签

```r
title(
	main= str,
	sub= str,
	xlab= str,
	ylab= str,

	col.main= str,
		# 也可以指定文本大小、字体、旋转角度、颜色
	col.sub= str,
	col.lab= str,
	cex.lab= str
)
```

####	`axis`

创建自定义坐标轴

```r
axis(
	side= int,
		# 坐标轴位置
		# 1-4：下、左、上、右
	at= c(int),
		# 需要绘制刻度线的位置
	labels= c(str),	
		# 刻度线对应labels
		# 默认为`at`中值
	pos= num,
		# 坐标轴线绘制位置坐标，即与另一坐标轴交点
	lty= num,
		# 线条类型
	col= str,
		# 线条、刻度颜色
	las= 0/2,
		# 标签平行、垂直坐标轴
	tck= num,
		# 刻度线长度，相对于绘图区域大小分数表示
		# 负值表示在图形外侧
		# `0`表示禁用刻度线
		# `1`表示绘制网格线
		# 默认`-0.01`
)
```

#####	`minor.tick`

次要刻度线

```r
library(Hmisc)
minor.tick(
	nx= int,
		# x轴次要刻度**区间**数
	ny= int,
	tick.ratio= num
		# 次要刻度线相较主要刻度线比例
)
```

####	`abline`

添加参考线

```r
abline(
	h= c(int),
		# 水平参考线位置（多根）
	v= c(int),
	lty= int,
	col= str,
)
```

####	`legend`

添加图例

```r
legend(
	location= c(x, y)/
		# 指定`x, y`坐标
		"bottom"/"bottomleft"/"left"/"topleft"/
		"top"/"topright"/"right"/"bottoright"/"center"/
		# 使用关键字指定图例位置
		locator(1),
		# 鼠标交互式指定
	inset= num,
		# `location`使用关键字指定位置，设置图例向图形内侧移动
		# 移动`num`绘图区域比例
	title= str,
		# 图例标题字符串
	legend= c(str),
		# 图例labels字符串向量
	col= c(col_str),
	pch= c(pch_int),
		# 若图例标识符号不同的点
	lwd= c(lwd_int),
	lty= c(lty_int),
		# 若图例标识宽度、样式不同的线
	fill= c(col_str),
		# 若图例标识颜色填充的盒型
)
```

####	`mtext`、`text`

-	`mtext`：向图形四个边界之一添加文本
-	`text`：向绘图区域内部添加文本

```r
text(
	location,
		# 同`legend`
	"text to place",
	pos= int,
		# 文本相对于`location`方向
		# 1-4：下、左、上、右
	offset= num,
		# 文本相对`location`偏移量
	cex= num,
	col= c(col_str),
	font= num,
)

mtext(
	"text to place",
	side= int,
		# 放置文本的边
		# 1-4：下、左、上、右
	line= num,
		# 向内、外移动文本
		# 越大文本越向外移动
	adj= int,
		# 1：文本坐下对齐
		# 2：文本右上对齐
	cex= num,
	col= c(col_str),
	font= num,
)

```

###	图形组合

####	`par`

```r
par(
	mfrow= c(nrows, ncols)
)
	# 设置按行填充的图形矩阵
par(
	mfcol= c(nrows, ncols)
)
	# 设置按列填充的图形矩阵
	# 然后按照的绘图顺序依次填充矩阵

par(
	fig= c(x1, x2, y1, y2)
)
	# 接下来图形在`fig`指定范围内绘制
```

####	`layout`

```r
layout(
	mat= matrix(int),
		# `mat`中元素值`n`表示第n个绘制的图形
		# 其在矩阵中所处的位置即其填充的位置
	widths= c(num),
		# 各列宽度值向量
		# 相对图形比例可以直接数值表示
		# 绝对数值可以通过函数`lcm`指定
	heights= c(num),
		# 各行高度值向量
```

##	R原始图表

###	条形图

```r
barplot(
	H,
	xlab,
	ylab,
	main,
	names.arg,
	col
)
```
####	一般条形图

```r
H <- c(7, 12, 28, 3, 41)
M <- c("Mar", "Apr", "May", "Jun", "Jul")

png(file = "barchart.png")
	# 设置图表文件名
barplot(
	H,
		# 各组数据大小
	names.arg = M,
		# 各组labels
	xlab = "Month",
		# x轴名
	ylab = "Revenue",
		# y轴名
	col = "blue",
		# 条形填充颜色
	main = "Revenue Chart",
		# 标题
	border = "red"
		# 条形边框颜色
)
	# 绘制图表
dev.off()
	# 保存图表
```

####	组合、堆积条形图

```r
colors <- c("green", "orange", "brown")
months <- c("Mar", "Apr", "May", "Jun", "Jul")
regions <- c("East", "West", "North")

values <- matrix(
	c(2,9,3,11,9,4,8,7,3,12,5,2,8,10,11),
	nrow = 3,
	ncol = 5,
	byrow = TRUE)
png(file = "barchart_stacked.png")
barplot(
	values,
	main = "total revenue",
	names.arg = months,
	xlab = "month",
	ylab = "revenue",
	col = colors)

legend(
	"topleft",
	regions,
	cex = 1.3,
	fill = colors)

dev.off()
```

###	饼图

```r
pie(
	x,
	labels,
	radius,
	main,
	col,
	clockwise
)
```

####	一般饼图

```r
x <- c(21, 62, 10, 53)
labels <- c("London", "New York", "Singapore", "Mumbai")
piepercents <- round(100 * x / sum(x), 1)
	# `round`取小数位数

png(file = "city.png")
pie(
	x,
	labels = piepercents,
	main = "city pie chart",
	col = rainbow(length(x))
)

legend(
	"topright",
	labels,
		# 饼图注解设为百分比，所以这里设置各组别labels
	cex = 0.8,
	fill = rainbow(length(x))
)

dev.off()
```

####	3D饼图

```r
library(plotrix)
x <- c(21, 62, 10, 53)
labels <- c("London", "New York", "Singapore", "Mumbai")

png(file = "3d_pie_charts.jpg")

pie3D(
	x,
	labels,
	explode = 0.1
	main = "pie charts of countries"
)

dev.off()
```

###	直方图

```r
hist(
	v,
	main,
	xlab,
	xlim,
	ylim,
	breaks,
		# 每个直方的宽度
	col,
	border
)
```

####	一般直方图

```r
v <- c(9,13,21,8,36,22,12,41,31,33,19)

png(file = "histogram.png")

hist(
	v,
	xlab = "weight",
	col = "green",
	border = "blue"
	xlim = c(0, 40),
	ylim = c(0, 5),
	breaks = 5
)

dev.off()
```
