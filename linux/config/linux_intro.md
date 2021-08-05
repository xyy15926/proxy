---
title: Linux介绍
categories:
  - Linux
  - Configuration
tags:
  - Linux
  - Installment
  - Configuration
date: 2019-07-31 21:11:52
updated: 2019-02-19 17:20:02
toc: true
mathjax: true
comments: true
description: Linux介绍
---

##	Linux版本

###	内核版本

内核版本号由3个数字组成`X.Y.Z-P`

-	`X`：主版本号，比较稳定，短时间不会改变
-	`Y`：次版本号，表示版本类型
	-	偶数：稳定版
	-	奇数：测试版
-	`Z`：发布号，数字越大，功能越完善
-	`P`：patch号

##	Linux分区

###	`/boot`引导分区目录

该分区（目录）存放系统内核、驱动模块引导程序，**需要**独立
分区

-	避免（根）文件系统损坏造成无法启动
-	避使用*lilo*引导时1024柱面问题（*Grub*无此问题）
-	方便管理多系统引导

####	`/boot`修复

进入grub模式后#todo

####	`/swap`分区目录

系统物理内存不足时，释放部分空间，其中数据被临时保存在
*swap*空间中

-	不是所有物理内存中交换的数据都会被放在交换空间中，有部分
	数据直接交换到文件系统

-	交换空间比内存慢

-	安装时，系统会尝试将交换分区安装到磁盘外端

-	有多个磁盘控制器时，在每个磁盘上都建立交换空间

-	尽量将交换空间安装在访问在频繁的数据区附近

-	交换空间大小一般设置为内存1-2倍

> - 不推荐为交换空间划分单独分区，可以使用交换文件作为交换
	空间，方便、容易扩展

#####	交换文件

```shell
$ dd if=/dev/zero of=/swapfile bs=1024 count=32000
	# 或
$ fallocate -l 32G /swapfile
	# 创建有连续空间的交换文件，大小为1024*32000=32G
$ chmod 600 /swapfile
	# 修改交换文件权限
$ mkswap /swapfile
	# 设置交换文件

$ /usr/sbin/swapon /swapfile
	# 激活上步创建的`/swapfile`交换文件
$ /usr/sbin/swapoff swapfile
	# 关闭交换文件
```

> - 不需要交换文件时可以直接`rm`删除
> - 可以在`fstab`文件中添加交换文件，自动挂载，格式参见
	`config_files`

###	`/`根分区目录

-	`/usr`：用户程序
-	`/sbin`：系统管理员执行程序
-	`/bin`：基本命令
-	`/lib`：基本共享库、核心模块
-	`/home`：用户目录
-	`/etc`：配置文件目录
-	`/opt`：附加应用程序包目录
-	`/mnt`：设备/文件系统挂载目录
-	`/dev`：设备
-	`/tmp`：临时文件
-	`/var`：可变信息区
	-	file spool
	-	logs
	-	requests
	-	mail
-	`/proc`：进程（映射）信息

