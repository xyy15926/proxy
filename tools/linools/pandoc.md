---
title: Pandoc介绍
tags:
  - Tools
  - LaTex
  - Markdown
  - pdf
categories:
  - Tools
date: 2020-07-12 18:09:40
updated: 2020-07-12 18:09:44
toc: true
mathjax: true
comments: true
description: Pandoc简介
---

##	Pandoc

Pandoc：将文本在不同标记语言之间相互转换的工具

-	Pandoc使用Haskell开发，最新版本Pandoc可以使用Haskell平台
	包管理器cabal安装

	```sh
	$ sudo apt install haskell-platform
	$ cabal update
	$ cabal install pandoc
	```

-	Pandoc支持的标记语言格式包括
	（具体可通过`--list-input-formats`查看）
	-	Markdown
	-	ReStructuredText
	-	HTML
	-	LaTeX
	-	ePub
	-	MS Word Docx
	-	PDF

-	Pandoc输入、输出文本
	-	可从输入、输出的文件扩展名推测输入、输出格式
	-	缺省输入格式为Markdown、缺省输出格式为html
	-	若未指明输入、输出文件，则读取、写入标准输入、输出
	-	文本默认为`utf-8`编码，否则可以用`iconv`转换文本
		编码进行输入、输出

##	Pandoc相关选项

###	基础选项

-	`-f`/`-t`：输入、输出格式
-	`-o`：输出文件
-	`--number-sections`/`-N`：为标题添加编号
-	`--verbose`：详细调试信息
	-	其中会给出资源文件目录`~/.pandoc`，存放模板等
-	`--log`：日志信息
-	`--file-scope`：分别转换每个文件
	-	指定多个输入文件时默认将多个文件拼接，空行分隔

###	信息选项

-	`--list-input-formats`/`--list-output-formats`：输入、
	输出格式
-	`--list-extensions[=FORMAT]`：列出Markdown扩展支持情况
	-	`<FORMAT>-<EXT>`可以增减格式中一个、多个扩展选项
-	`--list-highlight-languages`：语法高亮支持语言
-	`--list-highlight-styles`：语法高亮支持样式

###	模板选项

-	`-standalone`/`-s`：生成完整文件
	（仅对可生成片段的某些格式：html、LaTeX）
	-	默认生成文档片段
	-	采用相应内值模板生成完整文件
-	`--print-default-template=<FORMAT>`/`-D <FORMAT>`：输出
	对应格式的默认模板
-	`--template=<TPL>`：指定创建文档所需模板
-	`--css=<URL>`/`-c <URL>`：指定CSS样式表

###	自定义值传递

-	`--variable=<KEY[:<VAL>]>`/`-V <KEY[:<VAL>]>`：指定模板
	变量取值
-	`--metadata=<KEY[:<VAL>]>`/`-M <KEY[:<VAL>]>`：指定
	元数据字段取值
	-	元数据字段影响模板变量取值，同时影响底层文档元数据
-	`--metadata-file=<FILE>`：从YAML格式文件中设置元数据字段
	取值

###	PDF生成相关

-	`--toc`：生成目录
-	`--template=<TPL>`：编译使用的LaTeX模板，缺省为自带
-	`--latex-engine=<ENG>`：指定LaTeX引擎，需安装
	-	默认`pdflatex`对中文支持缺失，建议使用`xelatex`
-	`--highlight-style=<STY>`：代码块语法高亮
	-	自带高亮样式可通过`--list-highlight-styles`查看
	-	也可指定高亮样式文件
-	`--listings`：LeTeX文档中使用Listings包格式化代码块
-	`--biblatex`/`--natbib`：指定处理参考文献程序
-	`--bibliography=<FILE>`：设置文档元数据中参考文献信息

> - `-f markdown-implicit_figures`设置Markdown格式，指定
	图像保持原始位置，避免pdf中错位

###	HTML生成相关

-	`--self-contained`：将css、图片等所有外部文件压缩进html
	文件中

> - Markdown中使用html标记可以在转换为html后保留，可以据此
	设置转换后的样式

###	DOCX生成相关

-	`--reference-doc=<FILE>`：指定格式参考文件
	-	参考文件内容被忽略，就样式、文档属性被使用
-	`--print-default-data-file=reference.docx`：输出系统默认
	模板
-	`--extract-media=<DIR>`：提取文档中多媒体文件至文件夹，
	并在目标文件中设置对其引用

###	EPUB生成相关

-	`--epub-cover-image=<FILE>`
-	`--epub-metadata=<FILE>`
-	`--epub-embed-font=<FILE>`

###	数学公式渲染

-	`--mathjax[=<URL>]`
-	`--mathml`
-	`--katex[=<URL>]`

##	Pandoc模板

###	模板变量

模板变量：Pandoc模板中可以包含变量用于自定义模板

```conf
$title$						# 变量表示方法

$if(var)$					# 变量条件语句
X
$else$
Y
$endif$

$for(var)$					# 变量循环语句
X
$endfor$
```

-	变量赋值方式
	-	命令行参数提供
	-	文档元数据中查找


