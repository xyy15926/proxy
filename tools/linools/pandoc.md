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

-	`-f`/`-t`：输入、输出格式
-	`-o`：输出文件
-	`--verbose`：详细调试信息
	-	其中会给出资源文件目录`~/.pandoc`，存放模板等
-	`--log`：日志信息
-	`--list-input-formats`
-	`--list-outpu-formats`
-	`--list-extensions`：列出Markdown扩展支持情况
-	`--list-highlight-languages`：语法高亮支持语言
-	`--list-highlight-styles`：语法高亮支持样式

###	PDF生成相关

-	`--toc`：生成目录
-	`-N`：为标题添加编号
-	`--latex-engine=[]`：指定LaTeX引擎，需安装
	-	默认`pdflatex`对中文支持缺失，建议使用`xelatex`
-	`--template=[]`：编译使用的LaTeX模板，缺省为自带
-	`--highlight-style=[]`：代码块语法高亮
	-	自带高亮样式可通过`--list-highlight-styles`查看
	-	也可指定高亮样式文件

> - `-f markdown-implicit_figures`指定Markdown文本中图像保持
	原始位置，避免pdf中错位

###	HTML生成相关

-	`-s`：生成独立html文件
-	`--self-contained`：将css、图片等所有外部文件压缩进html
	文件中
-	`-c []`：指定css样式文件

> - Markdown中使用html标记可以在转换为html后保留，可以据此
	设置转换后的样式




