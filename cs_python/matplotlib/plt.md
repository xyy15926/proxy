---
title: Matplotlib.plot笔记
categories:
  - Python
  - Matplotlib
tags:
  - Python
  - Data Visualization
  - Matplotlib
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Matplotlib.plot笔记
---

##	图像元素

###	Figure

整个窗口（句柄）

```python
fig=plt.figure(num=None, figsize=None, dpi=None,
facecolor=None, edgecolor=None, frameon=True)
```

###	Axes

图像（子图）（句柄）

```python
None=plt.subplot(*arg, **karg)
	// 创建并激活axes
None=plt.subplot(num=int(row-col-index)
None=plt.subplot(row, col, index)

fig, axs=plt.subplots(rows=int, cols=int)

fig.add_subplot(rows=int, cols=int, index=int)

ax=fig.add_axes(rect=[left,buttom,width,height],
	projection='aitoff'/'hammer'...)
	// 指定地点、大小子图
	// `rect`参数是百分比
```

###	Axies

轴

###	说明

-	这三个图像元素都是以句柄形式操作，可以理解为每次操作
	都是在一个“容器”中操作，容器由plt管理，每次
	`plt.show()`将图像全部展示之后，容器被回收，创建新容器

-	因此`plt.show()`之后，之前容器中的元素更改无法影响到新容器

-	`plt`可以当作当前激活状态的axes元素的句柄使用

##	绘图

###	线图

```python

None=ax(*args, **kwargs)
	# `args`
		# [x]=range(len(y))/list
		# y=list
		# [fmt]=str

	# `kwargs`
		# data：类似df、dict的indexable对象，传递时x、y可用
			标签代替
		# linewidth
		# markersize
		# color
		# marker
		# linestyle

# examples
None=ax([x], y, [fmt], [x2], y2, [fmt2], **kwargs）
None=ax('xlabel', 'ylabel', data=indexable_obj)
None=ax(2d-array[0], 2d-array[1:])
```

###	柱状图

```python

list=ax.bar(*args, **kwargs)

	# `args`
		# x=list
		# height=num/list
		# [width]=num/list
		# [bottom]=num/list

	# `kwargs`
		# align='center'/'edge'
		# agg_filter=func
		# alpha=None/float
		# capstyle='butt'/'round'/'projecting'
#todo
```

###	饼图

```python
list=ax.pie(x, explode=None/list(%), labels=None/list,
	colors=None/list(color), autopct=None/str/func,
	pctdistance=0.6, shadow=False, labeldistance=1.1,
	startangle=None, radius=None/float,
	counterclock=True, textprops=None, center=(0,0),
	frame=False, rotatelables=False, hold=None,
	data=None/dict(param_above))
```

###	箱线图

```python
list=ax.boxplot(x, notch=False, sym=None/str,
	vert=True, whis=1.5/str/list)
```
