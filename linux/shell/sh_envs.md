---
title: Shell 环境变量
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Environment
date: 2019-07-31 21:10:52
updated: 2021-07-29 21:38:25
toc: true
mathjax: true
comments: true
description: Linux 环境变量
---

##	环境

###	`export`

显示、设置环境变量，使其可在shell子系统中使用

-	设置环境变量直接`$ ENV=value`即可，但是此环境变量
	不能在子shell中使用，只有`$ export ENV`导出后才可
-	`-f`：变量名称为函数名称
-	`-n`：取消导出变量，原shell仍可用
-	`-p`：列出所有shell赋予的环境变量


##	系统环境变量

###	NAME

-	`PATH`：用户命令查找目录
-	`HOME`：用户主工作目录
-	`SHELL`：用户使用shell
-	`LOGNAME`：用户登录名
-	`LANG/LANGUAGE`：语言设置
-	`MAIL`：用户邮件存储目录
-	`PS1`：命令基本提示符
-	`PS2`：命令附属提示符
-	`HISTSIZE`：保存历史命令记录条数
-	`HOSTNAME`：主机名称

> - `/etc/passwd`、`/etc/hostname`等文件中设置各用户部分
	默认值，缺省随系统改变

###	PATH

####	C PATH

-	`LIBRARY_PATH`：程序编译时，动态链接库查找路径
-	`LD_LIBRARAY_PATH`：程序加载/运行时，动态链接库查找路径

> - 动态链接库寻找由`/lib/ld.so`实现，缺省包含`/usr/lib`、
	`/usr/lib64`等
> > - 建议使用`/etc/ld.so.conf`配置替代`LD_LIBRARY_PATH`，
	或在编译时使用`-R<path>`指定
> > - 手动添加动态链接库至`/lib`、`/usr/lib`等中时，可能
	需要调用`ldconfig`生成cache，否则无法找到


