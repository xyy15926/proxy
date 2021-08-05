---
title: Shell 本地化
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Locale
date: 2019-07-31 21:10:52
updated: 2021-07-29 21:56:51
toc: true
mathjax: true
comments: true
description: Shell 本地化
---

##	本地化

###	字体

####	`fc-` 

-	`fc-list`：列出系统已安装字体

	```shell
	# 仅展示中文字体
	$ fc-list :lang=zh
	```

-	`fc-cache`：创建字体信息缓存文件

-	`mkfontdir/mkfontscale`：创建字体文件索引

-	字体安装
	-	将字体文件复制至字体文件夹
		-	`$HOME/.fonts`
		-	`/usr/share/fonts`
	-	`fc-cache` 更新缓存信息

