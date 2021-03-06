---
title: 用户、登陆
tags:
  - Linux
  - Shell命令
categories:
  - Linux
  - Shell命令
date: 2019-07-31 21:10:52
updated: 2019-06-21 00:49:03
toc: true
mathjax: true
comments: true
description: 用户、登陆
---

用户类型

-	超级用户：root用户
-	系统用户：与系统服务相关的用户，安装软件包时创建
-	普通用户：root用户创建，权限有限

###	显示登陆用户

###	`w`

详细查询已登录当前计算机用户

###	`who`

显示已登录当前计算机用户简单信息

###	`logname`

显示当前用户登陆名称

###	`users`

用单独一行显示当前登陆用户

###	`last`

显示近期用户登陆情况

###	`lasttb`

列出登陆系统失败用户信息

###	`lastlog`

查看用户上次登陆信息


##	用户、用户组

###	`newusers`

更新、批量创建新用户

###	`lnewusers`

从标准输入中读取数据创建用户

###	`userdel`

删除用户账户

###	`groupdel`

删除用户组

###	`passwd`

设置、修改用户密码

###	`chpassws`

成批修改用户口令

###	`change`

更改用户密码到期信息

###	`chsh`

更改用户账户shell类型

###	`pwck`

校验`/etc/passwd`和`/etc/shadow`文件是否合法、完整

###	`grpck`

验证用户组文件`/etc/grous/`和`/etc/gshadow`完整性

###	`newgrp`

将用户账户以另一个组群身份进行登陆

###	`finger`

用户信息查找

###	`groups`

显示指定用户的组群成员身份

###	`id`

显示用户uid及用户所属组群gid

###	`su`

切换值其他用户账户登陆

###	`sudo`

以superuser用户执行命令

-	Archlinux中需要自行安装
-	配置文件为`/etc/sudoers`

###	`useradd/adduser`

创建用户账户（`adduser`：`useradd`命令的符号链接）

-	`-c`：用户描述
-	`-m`：创建用户目录
-	`-d`：用户起始目录
-	`-g`：指定用户所属组
-	`-n`：取消建立以用户为名称的群组
-	`-u`：指定用户ID
-	`-s`：指定用户登录shell
	-	缺省为空，默认值应该是`/bin/sh`，很多发行版会设置
		其为`/bin/bash`
	-	查看`$SHELL`环境变量查看当前shell
	-	文件`/etc/shells`包含支持shell类型

###	`usermod`

修改用户

-	`-e`：账户有效期限
-	`-f`：用户密码过期后关闭账号天数
-	`-g`：用户所属组
-	`-G`：用户所属附加组
-	`-l`：用户账号名称
-	`-L`：锁定用户账号密码
-	`-U`：解除密码锁定

###	`groupadd`

新建群组

-	`-g`：强制把某个ID分配给已经存在的用户组，必须唯一、非负
-	`-p`：用户组密码
-	`-r`：创建系统用户组

###	`groupmod`

-	`-g`：设置GID
-	`-o`：允许多个用户组使用同一个GID
-	`-n`：设置用户组名

##	登陆、退出、关机、重启

###	`login`

登陆系统

###	`logout`

退出shell

###	`exit`

退出shell（常用）

###	`rlogin`

远程登陆服务器

###	`poweroff`

关闭系统，并将关闭记录写入`/var/log/wtmp`日志文件
###	`ctrlaltdel`

强制或安全重启服务器

###	`shutdown`

关闭系统

###	`halt`

关闭系统

###	`reboot`

重启系统

###	`init 0/6`

关机/重启


