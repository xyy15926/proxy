---
title: Shell 常用工具
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Tool
  - Datetime
  - Network
date: 2019-07-31 21:10:52
updated: 2021-08-10 16:39:26
toc: true
mathjax: true
comments: true
description: Shell 常用工具
---

##	网络传输

###	`wget`、`curl`

-	`wget`：从指定 URL 下载文件
	-	`-b`：后台执行
	-	`-c`：继续执行上次任务
	-	`-r`：递归下载
	-	`-O`：下载文件重命名
	-	`-nc`：不覆盖同名文件
	-	`-nv`：不显示执行过程
	-	`-P`：指定下载路径
	-	`--no-check-certificate`：跳过证书检查过程

-	`curl`：综合文件传输工具

##	日期时间

###	`date`

```sh
date -d <time> "+<format>"
```

-	`date`：显示、设置系统日期时间
	-	`-d`：指定时间，缺省今天
	-	`+<format>`：指定输出格式
		-	`%Y-%m-%d %h-%M-%S`：年月日时（24时）分秒
		-	`%a/%A`：星期缩写、完整
		-	`%b/%B`：月份缩写、完整
		-	`%D`：`MM/DD/YY`
		-	`%F`：`YYYY-MM-DD`

##	命令总结

-	日期、时间
	-	`cal`：显示日历信息
	-	`date`：显示、设置系统日期时间
	-	`hwclock`：查看、设置硬件时钟
	-	`clockdiff`：主机直接测量时钟差
	-	`rdate`：通过网络获取时间
	-	`sleep`：暂停指定时间

-	数值计算
	-	`bc`：任意精度计算器
	-	`expr`：将表达式值打印到标准输出，注意转义

-	网络工具
	-	`wget`：从指定 URL 下载文件
	-	`curl`：综合文件传输工具

-	获取命令系统帮助
	-	`help`：查看 shell 内建命令帮助信息 **（*bash* 内建）**
	-	`man`：显示在线帮助手册（常用）
	-	`info`：info 格式的帮助文档


