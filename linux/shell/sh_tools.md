---
title: Shell 常用工具
tags:
  - Linux
  - Shell
categories:
  - Linux
  - Shell
  - Tool
date: 2019-07-31 21:10:52
updated: 2021-07-29 21:47:37
toc: true
mathjax: true
comments: true
description: Shell 常用工具
---

##	获取命令系统帮助

###	`help`

重看内部shell命令帮助信息（常用）

###	`man`

显示在线帮助手册（常用）

###	`info`

info格式的帮助文档

##	打印、日期、时间

###	`echo`

输出至标准输出

###	`env`

打印当前所有环境变量

###	`cal`

显示日历信息

###	`date`

`date`：显示、设置系统日期时间

```shell
date -d <time> "+<format>"
```

-	`-d`：指定时间，缺省今天
-	`+`：指定输出格式
	-	`%Y-%m-%d %h-%M-%S`：年月日时（24时）分秒
	-	`%a/%A`：星期缩写、完整
	-	`%b/%B`：月份缩写、完整
	-	`%D`：`MM/DD/YY`
	-	`%F`：`YYYY-MM-DD`

###	`hwclock`

查看、设置硬件时钟

###	`clockdiff`

主机直接测量时钟差

###	`rdate`

通过网络获取时间

###	`sleep`

暂停指定时间


##	数值计算

###	`bc`

任意精度计算器

###	`expr`

将表达式值打印到标准输出，注意转义


