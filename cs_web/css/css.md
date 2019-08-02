---
title: CSS
tags:
  - Web
categories:
  - Web
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: CSS
---

##	CSS属性

-	`display`：元素展示
	-	`inline`：默认，内联元素
	-	`flex`：弹性容器
	-	`none`：不显示
	-	`block`：块级元素，元素前后有换行符
	-	`inline-block`：行内块元素
	-	`list-item`：列表元素
	-	`run-in`：根据上下文作为块级元素或内联元素
	-	`table`：表格元素`<table>`，前后有换行符
		-	`inline-table`：内联表格元素，前后无换行符
		-	`table-row-group`
		-	`table-row`
		-	`table-header-group`
		-	`table-footer-group`
		-	`table-column-group`
		-	`table-column`
		-	`table-cell`
		-	`table-caption`
	-	`inherit`


###	弹性容器Flex

####	`flex`

**弹性容器**内元素空间分配

-	`flex-grow flex-shrink flex-basis`简写，只给出一个值时
	相当于`flex-grow`
-	`auto`：同`1 1 auto`
-	`none`：同`0 0 auto`
-	`initial`：初始值，同`0 0 auto`
-	`inherit`：

####	`flex-grow`

（弹性容器内）元素相对于其他元素项目扩展的量

-	`auto`
-	`inherit`
-	`{num}`：相对于其他元素的扩展量，默认为0，应该是按照比例分

####	`flex-shrink`

（弹性容器内）元素相对于其他元素收缩的量

####	`flex-basis`

（弹性容器内）元素基础长度

-	`auto`
-	`inherit`
-	具体长度（`%, px, em`为单位）

####	`flex-flow`

-	`flex-direction flex-wrap`
-	`initial`
-	`inherit`

####	`flex-direction`

弹性容器内元素方向

-	`row`：默认，灵活元素水平排列
-	`row-reverse`：灵活元素水平反向排列
-	`column`：灵活元素竖直排列
-	`column-reverse`：灵活元素竖直反向排列
-	`initial`
-	`inherit`

####	`flex-wrap`

弹性容器内元素是否拆行/列

-	`nowrap`：默认，灵活项目不拆行/列
-	`wrap`：灵活项目必要时拆行/列
-	`wrap-reverse`：灵活项目必要时反向拆行/列
-	`initial`
-	`inherit`

###	弹性容器Align

####	`align-items`

**弹性容器**内侧轴（竖直方向）对齐各项元素，适合含有单行元素

-	`stretch`：默认，元素拉伸对齐，元素大小确定时同
	`flex-start`效果
-	`center`：元素堆叠在容器中心
-	`flex-start`：元素堆叠向容器开头
-	`flex-end`：元素堆叠向容器结尾
-	`baseline`：元素位于容器基线上
-	`initial`：初始值
-	`inherit`：从父类继承

####	`align-content`

和`align-items`有相同的功能，适合多行元素
（这个好像作用单位是**弹性容器**的行）

-	`stretch`：默认，元素拉伸对齐，元素大小确定时同
	`flex-start`效果
-	`center`：元素堆叠在容器中心
-	`flex-start`：元素堆叠向容器开头
-	`flex-end`：元素堆叠向容器结尾
-	`space-between`：元素向容器两端堆叠，元素间保持间距
-	`space-around`：类似`space-between`，但元素于容器
	边界直接也有间距
-	`initial`：初始值
-	`inherit`：从父类继承

####	`align-self`

设置**弹性容器**内元素本身侧轴方向上的对齐方式

-	`auto`：默认，继承父容器`align-items`属性，没有
	父容器则为`stretch`
-	`stretch`：元素拉伸对齐，元素大小确定时同
	`flex-start`效果
-	`center`：元素堆叠在容器中心
-	`flex-start`：元素堆叠向容器开头
-	`flex-end`：元素堆叠向容器结尾
-	`baseline`：元素位于容器基线上
-	`initial`：初始值
-	`inherit`：从父类继承

####	`justify`

**弹性容器**内水平排列各项元素

-	`flex-start`：默认，元素位于容器开头
-	`flex-end`：元素堆叠向容器结尾
-	`center`：元素堆叠在容器中心
-	`space-between`：元素向容器两端堆叠，元素间保持间距
-	`space-around`：类似`space-between`，但元素于容器
	边界直接也有间距
-	`initial`：初始值
-	`inherit`：从父类继承

