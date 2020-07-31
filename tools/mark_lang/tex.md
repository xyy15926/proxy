---
title: TeX、LaTeX、XeLaTeX
tags:
  - TeX
  - LaTeX
  - MarkupLang
categories:
  - Tools
  - MarkupLang
date: 2020-07-12 10:53:21
updated: 2020-07-12 10:53:31
toc: true
mathjax: true
description: TeX、LaTeX、XeLateX等介绍
---

##	TeX

TeX：由Donald Knuth开发的高质量排版系统

-	TeX的范畴包括标记语法（命令）、具体实现（引擎），在不同
	场合含义不同，类似语言、编译器

-	初版TeX包含300左右命令，随后Donald添加Plain TeX扩展包
	定义了600左右常用宏命令

> - TeX中将标记关键字称为命令，其实关键字确实类似命令，可能
	系统对应命令、语言对应关键字

##	TeX宏包

-	`*.cls`类文件：定义文档结构
-	`*.sty`包/样式文件：提供类未包括的任何东西
	-	对类文件功能的补充：提供从属于类文件的功能
	-	对类文件功能的修改：改变类文件的风格

###	基础排版宏包

-	LaTeX：Leslie Lamport设计的更高层次、更抽象的排版格式
	-	包含以TeX命令为基础的一系列高层、抽象宏命令，包括
		`\section`、`\usepackage`等
	-	用户可以使用模板而不必决定具体排版、打印，使用更加
		方便
	-	是学界事实上的标准排版格式

-	ConTeXt：Pragma-ADE公司设计的文档制造格式
	-	为TeX提供对先进打印特性的易用接口
	-	生成的编译文件更美观，适合专业印刷行业使用
	
-	TeXinfo：FSF（Free Software Foundation）设计的格式
	-	是Linux系统的标准文档系统

###	中文排版宏包

-	xeCJK：在XeTeX引擎下处理中日韩文字断行、标点调整、字体
	选择的基础性宏包
	-	基础从CCT、CJK以来的标点禁则、标点压缩、NFSS字体补丁
	-	实际上仅对简体中文处理机制比较完全
	-	XeTeX本身已经能够大概处理中文文档的排版，但在某些
		细节部分仍然可以改进

	> - 类似的宏包还有：zhspacing、xCJK、xCCT

-	CTeX：提供了编写中文文档时常用的宏命令

###	其他宏包

-	AMS-TeX/AMS-LaTeX：AMS（American Mathematical Society）
	设计的格式
	-	提供了额外的数学字体、多行数学表述排版

##	TeX引擎

-	pdfTex：将TeX文本编译输出为pdf格式的引擎实现
	-	最初TeX/LaTeX实现将TeX编译为DVI格式，然后方便转换为
		PostScript文件用于打印
	-	而pdf格式超越PostScript格式成为更流行的预打印格式
	-	因此事实上，现在LaTeX发行版中包含4个部分：TeX、
		pdfTeX、LaTeX、pdfLaTeX

-	XeTeX：扩展了TeX的字符、字体支持的引擎实现
	-	最初TeX/LaTeX仅仅支持英语数字、字母
	-	XeTeX提供了对Unicode字符和TrueType/OpenType字体的
		直接支持

-	LuaTeX：将TeX扩展为更sensible的编程语言的实现

##	TeX发行版

###	TeX Live

TeX Live：TUG（TeX User Group）发布、维护的TeX系统

-	TeX Live包含
	-	与TeX系统相关的各种程序
		-	pdfTeX
		-	XeTeX
		-	LuaTeX
	-	编辑查看工具
		-	DVIOUT DVI Viewer
		-	PS View
		-	TeXworks
	-	常用宏包
		-	LaTeX
	-	常用字体
	-	多个语言支持

> - TeX Live官网：<https://tug.org/texlive>

###	MiKTeX

MiKTeX：Christian Schenk开发的在MSWin下运行文字处理系统

> - MiKTeX官网：<http://miktex.org>

###	CTeX

CTeX：CTeX学会开发，将MiKTeX及常用应用封装

-	集成WinEdt编辑器
-	强化了对中文的处理


