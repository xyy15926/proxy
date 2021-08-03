---
title: 字体杂记
categories:
  - CS
  - Character
tags:
  - CS
  - Character
  - Font
date: 2019-03-21 17:27:37
updated: 2021-08-02 16:27:55
toc: true
mathjax: true
comments: true
description: 字体杂记
---

##	字体文件

-	*TTF*：*Truetype Font*
	-	苹果公司创建，*Mac* 和 *Win* 上最常用的字体文件格式
	-	由数学表达式定义、基于轮廓的字体文件
	-	保证了屏幕、打印输出的一致性，同时也可以和向量字体一样随意缩放

-	*TTC*：*TrueType Collection*
	-	微软开发的新一代字体文件，多个 *TTF* 文件合并而成
	-	可以共用字体笔画文件，有效的节约空间
	-	兼容性不如 *TTF*，有些应用无法识别

-	*FON*
	-	*Win* 上最小的字体文件格式
	-	固定大小的位图格式，难以调整字体大小

##	字体分类

###	西文字体分类

> - 西文字体分类方法有很多种，但是太学术，不常用，常用分类的可以看计算机字体族

-	*Thibaudeau* 分类法
	-	法国字体排印师 *Francis Thibaudeau* 于 1921 年提出
-	*Vox-ATypl* 分类法
	-	*Maximilien Vox* 于1954年提出，是比较早、基础、业内有过影响力的分类法
-	*Fontshop* 自家的分类法
	-	在已有的思路的基础上，基于字体开发的独特的分类法
	-	适合网上搜索字体，网罗了超过 15 万字体
-	*Linotype* 提供的3种分类法
	-	*by category + by usage+ by theme*
	-	后 2 者是面向一般字体用户，重视字体的用途来合理分类

###	中文字体分类

> - 中文字体则没有一个明确的分类体系，仅能大概分类

-	宋体（明体）：最能代表汉字风格的印刷字头
-	仿宋：相当于雕版时代的魏碑体
-	楷体：标准化的楷书，毛体书法的产物
-	黑体：汉字在西方现代印刷浪潮冲击下的产物
-	圆体：海外地区率先开发、使用

###	计算机字体分类

-	*serif*：有衬线字体
	-	特点
		-	笔画有修饰，末端向外展开、尖细或者有实际衬线
		-	文字末端在小号字体下容易辨认，大号可能模糊或有锯齿
	-	例
		-	*Times New Roman*、*MS Georgia*、*DejaVu Serif*
		-	*宋体*、*仿宋*
	-	两种衍生字体
		-	*petit-serif*：小衬字体，可以当作无衬线
		-	*slab-serif*：雕版衬线，末端变化非常明显

-	*san-serif*：无衬线字体
	-	特点
		-	末端笔画清晰，带有一点或没有向外展开、交错笔画
		-	与*serif*相比，字体较小时可能难以分辨、串行（阅读）
	-	举例
		-	*MS Trebuchet*、*MS Arial*、*MS Verdana*
		-	*黑体*、*圆体*、*隶书*、*楷体*

-	*monospace*：等宽字体
	-	特点
		-	每个字形等宽，因此常作为终端所用字体
	-	举例
		-	*Courier*、*MS Courier New*、*Prestige*
		-	多数中文字体（中文字体基本都等宽）

-	*cursive*：手写字体
	-	举例
		-	*Caflisch Script*、*Adobe Poetica*
		-	xx手写体、xx行草 

-	*fantasy*：梦幻字体（艺术字）
	-	举例
		-	*WingDings*、*WingDings2*、*Symbol*
		-	各种奇怪名字的字体

> - *serif*、*san-serif* 是西文字体的两大分类
> - 而后应该是计算机的出现带来的*monospace*的兴起
> - 最后面两种在正式场合中不常用

