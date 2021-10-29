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
updated: 2021-09-01 16:57:44
toc: true
mathjax: true
description: 
---

##	用户

###	`useradd/adduser`

-	`useradd`、`adduser`：创建用户账户
	-	`adduser` 是 `useradd` 命令的符号链接

-	选项参数
	-	`-c`：用户描述
	-	`-m`：创建用户目录
	-	`-d`：用户起始目录
	-	`-g`：指定用户所属组
	-	`-n`：取消建立以用户为名称的群组
	-	`-u`：指定用户 *UID*
	-	`-s <SHELL>`：指定用户登录 Shell
		-	缺省为空，默认值为 `/bin/sh`，很多发行版会设置其为 `/bin/bash`
		-	文件 `/etc/shells` 包含支持 Shell 类型

###	`usermod`

-	`usermod`：修改账户属性

-	选项参数
	-	`-e`：账户有效期限
	-	`-f`：用户密码过期后关闭账号天数
	-	`-g`：用户所属组（主群组）
	-	`-G`：用户所属附加组
	-	`-l`：用户账号名称
	-	`-L`：锁定用户账号密码
	-	`-U`：解除密码锁定

####	`/etc/passwd`

-	`/etc/passwd`：存储系统用户信息
	-	每行为存储单个用户信息的一条记录

-	每行格式、字段含义

	```cnf
	<USERNAME>:<PWD>:<UID>:<GID>:<COMMENT>:<HOME>:<LOGIN-CMD>
	```

	-	`PWD`：在 `/etc/passwd` 文件中以 `x` 占位，不实际存储密文
	-	`COMMENT`：账户注释
	-	`LOGIN-CMD`：用户登录之后执行的命令，一般将启动一个 Shell 程序

###	`su`

-	`su`：切换账户

-	选项参数
	-	`-`/`-l`/`--login`：像 *login-shell* 一样启动 Shell
		-	初始化用户环境，否则维持当前环境
	-	`-c <CMD>`/`--command=<CMD>`：使用 `-c` 选项向 Shell 传递命令
	-	`--session-command=<CMD>`：类似 `-c`，但不开启新的会话
	-	`-g <GROUP>`/`--group=<GROUP>`：指定初始群组，仅对 root 用户有效
	-	`-G <GROUP>`/`--supp-group=<GROUP>`：指定附加群组，若主群组未指定，首个附加群组将作为主群组
	-	`-s <SHELL>`/`--shell=<SHELL>`：使用特定 Shell

###	`passwd`

-	`passwd`：更改、设置账户密码

-	选项参数
	-	缺省：修改账户密码，缺省修改当前账户
	-	`-d <USERNAME>`：删除密码
	-	`-f <USERNAME>`：强制账户下次登录必须修改密码
	-	`-w <DAYS>`：账户口令失效警告提前天数
	-	`-k <USERNAME>`：过期则需要重设密码
	-	`-l <USERNAME>`：停止账户使用
	-	`-S <USERNAME>`：显示密码信息
	-	`-u <USERNAME>`：启用被停止账户
	-	`-n <TIME>`：指定密码最短有效期，即多久后可修改
	-	`-x <TIME>`：指定密码最长有效期
	-	`-g <GROUPNAME>`：修改群组密码
	-	`-i <TIME>`：设置账户密码过期后失效期

####	`/etc/shadow`

-	`/etc/shadow`：用户信息加密文件
	-	仅 root 用户有读权限
	-	每行代表一个用户信息

-	每行格式、字段含义

	```cnf
	<USERNAME>:<PWD>:<LAST-CHANGE>:<CHNAGE-INTERVAL>:<VALID>:<WARN>:<BUFFER>:<EXPIRE>:<RESERVED>
	```

	-	`PWD`：*SHA512* 加密密码
		-	密码项为 `!!`、`*` 表示用户无密码，不能登录
	-	`LAST-CHANGE`：自 *1/1/1970* 起，密码被修改的天数
	-	`CHANGE-INTERVAL`：密码修改间隔，即多少天后允许被修改，`0` 为任何时候可修改
	-	`VALID`：密码有效期，即多少天之后需修改
	-	`WARN`：密码过期前多少天提示用户，`-1` 无提示
	-	`BUFFER`：密码过期后可用天数，`-1`、空表示永不禁用
	-	`EXPIRE`：自 *1/1/1970* 起，账号失效天数，空表示永不禁用
	-	`RESERVED`：预留字段

###	`sudo`

-	`sudo`：以其他账户身份执行命令
	-	`$ sudo -h | -K | -k | -V`
	-	`$ sudo -v <OPS>`：更新时间戳
	-	`$ sudo -l <OPS>`：列出用户权限
	-	`$ sudo -e <OPS>`：编辑文件

-	选项参数
	-	`-A`/`--askpass`：使用帮助程序进行密码提示
	-	`-b`/`-background`：后台执行命令
	-	`-B`/`--bell`：提示时响铃
	-	`-C`/`--close-from=<NUM>`：关闭所有文件描述符大于 `NUM` 的文件
	-	`-E`/`--preserve-env=[ENV]`：执行命令时保留（指定）环境变量
	-	`-e`/`--edit`：编辑文件而不是执行命令
	-	`-g`/`--group=<GROUP>`：以指定群组名（GID）执行命令
	-	`-H`/`--set-home`：将 `$HOME` 设置为目标主目录
	-	`-h`/`--host=<HOST>`：在 `HOST` 上执行命令
	-	`-i`/`--login`：以指定用户身份运行 *login-shell*
	-	`-K`/`--remove-timestamp`：完全移除时间戳文件
	-	`-k`/`--reset-timestamp`：无效化时间戳文件
	-	`-l`/`--list`：列出用户权限、特定命令权限，两次使用得到详细信息
	-	`-n`/`--non-interactive`：非交互模式，没有提示符
	-	`-p`/`--prompt=<PROMPT>`：使用特定密码提示器
	-	`-P`/`--preserve-groups`：保留组向量
	-	`-r`/`--role=role`：用特定角色创建 SEL 安全上下文
	-	`-S`/`--stdin`：从标准输入中读取密码
	-	`-s`/`--shell`：以指定账户运行 Shell
	-	`-t`/`--type=<TYPE>`：创建特定类型的 SEL 安全上下文
	-	`-T`/`--command-timeout=<TIMEOUT>`：命令执行超时
	-	`-U`/`--other-user=<USER>`：列出用户权限
	-	`-u`/`--user=<USER>`：以指定账户名、UID 执行命令
	-	`-v`/`--validate`：不执行命令情况下更新用户时间戳

####	`/etc/sudoers`

-	`/etc/sudoer`：`sudo` 配置文件
	-	赋予用户、组`sudo`权限

##	群组

###	`groupadd`

-	`groupadd`：新建群组

-	选项参数
	-	`-g`：强制把某个 *GID* 分配给已经存在的用户组，必须唯一、非负
	-	`-p`：用户组密码
	-	`-r`：创建系统用户组

###	`groupmod`

-	`groupmod`：修改用户群组

-	选项参数
	-	`-g`：设置 *GID*
	-	`-o`：允许多个用户组使用同一个 *GID*
	-	`-n`：设置用户组名

####	`/etc/group`

-	`/etc/group`：群组账号信息文件
	-	每行代表一个群组信息

-	各行格式、字段含义

	```cnf
	<GROUPNAME>:<PWD>:<GID>:<USERS>
	```

	-	`PWD`：组密码，空、`*`、`x`、`!` 表示未设置密码
	-	`GID`：群组 ID
		-	系统群组：安装 Linux 以及部分服务型程序时自动设置的群组，GID < 500
		-	私人组群：由 root 新建的群组，GID > 500
	-	`USERS`：组成员列表
		-	不同用户之间用 `,` 分隔，可能是用户主组、附加组

####	`/etc/gshadow`

-	`/etc/gshadow`：群组信息加密文件
	-	每行代表一个用户信息

-	各行格式、字段含义

	```cnf
	<GROUPNAME>:<PWD>:<ADMIN>:<USERS>
	```

	-	`PWD`：加密组密码，空、`*`、`x`、`!` 表示未设置密码
	-	`ADMIN`：组管理员
	-	`USERS`：组成员列表

###	`newgrp`

-	`newgrp`：登录至其他群组

-	选项参数
	-	`-`：用户环境重新初始化，否则维持当前环境

##	命令总结

-	用户类型
	-	超级用户：root 用户
	-	系统用户：与系统服务相关的用户，安装软件包时创建
	-	普通用户：root 用户创建，权限有限

-	用户信息
	-	`w`：详细查询已登录当前计算机用户
	-	`who`：显示已登录当前计算机用户简单信息
	-	`logname`：显示当前用户登陆名称
	-	`users`：用单独一行显示当前登陆用户
	-	`last`：显示近期用户登陆情况
	-	`lasttb`：列出登陆系统失败用户信息
	-	`lastlog`：查看用户上次登陆信息

-	用户、用户组
	-	`newusers`：更新、批量创建新用户
	-	`lnewusers`：从标准输入中读取数据创建用户
	-	`userdel`：删除用户账户
	-	`groupdel`：删除用户组
	-	`passwd`：设置、修改用户密码
	-	`chpassws`：成批修改用户口令
	-	`change`：更改用户密码到期信息
	-	`chsh`：更改用户账户shell类型
	-	`pwck`：校验 `/etc/passwd` 和 `/etc/shadow` 文件是否合法、完整
	-	`grpck`：验证用户组文件 `/etc/grous/` 和 `/etc/gshadow` 完整性
	-	`newgrp`：将用户账户以另一个组群身份进行登陆
	-	`finger`：用户信息查找
	-	`groups`：显示指定用户的组群成员身份
	-	`id`：显示用户 *uid* 及用户所属组群 *gid*
	-	`su`：切换值其他用户账户登陆
	-	`sudo`：以 *superuser* 用户执行命令
	-	`useradd/adduser`：创建用户账户

-	登陆、退出、关机、重启
	-	`login`：登陆
	-	`logout`：登出
	-	`exit`：退出 Shell
	-	`rlogin`：远程登陆服务器
	-	`poweroff`：关闭系统，并将关闭记录写入 `/var/log/wtmp` 日志文件
	-	`ctrlaltdel`：强制或安全重启服务器
	-	`shutdown`：关闭系统
	-	`halt`：关闭系统
	-	`reboot`：重启系统
	-	`init 0/6`：关机/重启

