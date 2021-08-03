---
title: Shell 用户配置
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Profile
  - RC
date: 2021-07-29 21:21:30
updated: 2021-07-29 21:58:55
toc: true
mathjax: true
description: 
---

##	用户、组

###	用户信息

####	`/etc/passwd`

用户属性

-	用户名
-	口令：在`/etc/passwd`文件中以`x`显示
-	user-id：UID，用户标识
	-	linux事实上不直接处理用户名，只识别UID
	-	UID对用户是唯一的
-	group-id：GID，用户的默认组标识
	-	用户可以隶属多个不同的组，但是只有一个默认组，即用户
		创建文件默认隶属的组
-	描述：用户描述
-	用户主目录：用户登陆目录
-	登陆shell：用户登陆系统使用的shell
	-	缺省为空，可能是`/bin/sh`

####	`/etc/shadow`

用户口令属性

-	用户名
-	加密的密码
-	自1/1/1970起，密码被修改的天数
-	密码将被允许修改之前的天数（`0`为在任何时候可修改）
-	系统强制用户修改为新密码之前的天数（`1`永远不能修改）
-	提前警告用户密码过期前天数（`-1`无警告）
-	禁用用户账户在密码过期后天数（`-1`永不禁用）
-	用户被禁用天数（`-1`被启用）

####	`/etc/group`

群组账号信息文件

-	群组名
-	密码：以`x`显示
-	群组ID（GID）
	-	系统群组：安装Linux以及部分服务型程序时自动设置的
		群组，GID<500
	-	私人组群：由root新建的群组，默认GID>500
-	附加用户列表

####	`/etc/gshadow`

群组口令信息文件

####	`/etc/sudoers`

`sudo`配置文件

-	根据配置文件说明取消注释即可赋予用户、组`sudo`权限

###	用户配置文件

-	login-shell：用户登陆（创建session）时的shell模式，该
	模式下shell会自动执行*profile*文件
-	subshell：用户登陆后载入的shell的模式，该模式下shell会
	自动执行*rc*文件

####	*profile*

-一般在login-shell模式会执行一次，从名称看起来更像是
**用户配置**

-	全局、被所有用户默认执行的文件在`/etc`目录下，用户个人
	*profile*在用户目录

-	*^profile$*是所有类型shell（bash、zsh、ash、csh）都会
	执行

-	不同类型的shell可能有其特定的*profile*文件，如：
	`/etc/bash_profile`、`~/.bash_profile`，不过不常见
	（可以理解，毕竟是**用户配置**）

-	有的发行版本（ubuntu）还有有`/etc/profile.d`文件夹，在
	`/etc/profile`中会设置执行其中的配置文件

####	*rc*

*rc*应该是*run command*的简称，在每次subshell模式会执行，从
名称看起来更像是**shell配置**（很多应用配置文件*rc*结尾）

-	全局、被所有用户执行的文件在`/etc`目录下，用户个人*rc*
	则在用户目录

-	应该是因为*rc*本来就是对shell的配置文件，所以是不存在
	通用的*^rc$*配置的，最常用的bash对应就是`~/.bashrc`、
	`~/bash.bashrc`

####	总结

-	其实*rc*也会在用户登陆时执行
	-	login-shell会立刻载入subshell？
	-	*profile*里设置了立刻调用？

-	应该写在*profile*里的配置
	-	shell关系不大、更像是用户配置，如：特定应用环境变量
	-	不需要、不能重复执行，因为*rc*在用户登录时已经执行
		过一次，launch subshell时会重复执行，如：
		`export PATH=$PATH:xxxx/bin/`

-	应该写在*rc*里的配置
	-	和shell关系紧密的shell配值，如：alias
	-	在用户登陆后会该边，需要在每次launch subshell时执行
		的配置

-	配置文件执行顺序
	-	没有一个确定顺序，不同的linux发行版本有不同的设置，
		有的还会在脚本中显式写明相互调用，如：`/etc/profile`
		中调用`/etc/bashrc`，`~/.bashrc`调用`/etc/bashrc`
	-	但是可以确认的是`/etc/profile`一般是第一个被调用，
		`~/.xxxxrc`、`/etc/xxxxxrc`中的一个最后调用

-	还有一些其他配置文件
	-	`~/.bash_logout`：退出bash shell时执行

-	对于wsl，可能是因为将用户登陆windows视为create session，
	`~/.profile`好像是不会执行的

####	`/etc/environment`

系统在登陆时读取第一个文件

-	用于所有为所有进程设置环境变量
-	不是执行此文件中的命令，而是根据`KEY=VALUE`模式的
	代码，如：`PATH=$PATH:/path/to/bin`

##	用户命令

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

