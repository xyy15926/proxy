---
title: Linux 文件系统命令
categories:
  - Linux
  - File System
tags:
  - Linux
  - Kernel
  - File System
  - Ext2/3
  - Ext4
  - Commands
date: 2021-07-29 16:24:11
updated: 2021-08-25 18:52:39
toc: true
mathjax: true
description: 介绍 Linux 文件系统中常用命令
---

##	目录、文件操作

###	`du`、`df`

> - <https://www.junmajinlong.com/linux/du_df>

-	`du`：

-	`du` 选项参数
	-	`-a`/`--all`：显示 **所有后代** 各文件、文件夹大小（缺省仅显示文件夹）
	-	`-c`/`--total`：额外显示总和
	-	`-s`/`--summarize`：仅显示总和
	-	`-d`/`--max-depth=<num>`：超过 `num` 深度之后仅显示总和
	-	`-S`/`--separate-dirs`：文件夹大小不包括子文件夹大小
	-	`-b`/`--bytes`：以 byte 为单位
	-	`-k`/`--kilobytes`：以 KB 为单位
	-	`-m`/`--megabytes`：以 MB 为单位
	-	`-h`/`--human-readable`：自动转换适合单位
	-	`-H`/`--si`：同 `-h`，但是单位换算以 1000 为进制
	-	`-x`/`--one-file-system`：忽略其他遇到的文件系统
	-	`-D`/`--dereference-args`：仅计算命令行中出现符号链接源文件
	-	`-L`/`--dereference`：计算所有符号链接源文件
	-	`-X`/`--exclude=<ptn>`：排除匹配 `ptn` 的文件
		-	`--exclude-from=<file>`：从 `file` 读取需排除文件
	-	`--exclude=[dir/file]`：掠过指定目录、文件
	-	`-l`/`--count-links`：重复计算硬链接

###	`umask`

-	`umask`：查看、设置权限掩码，默认`0000`
	-	`$ umask`：打印当前权限掩码
	-	`$ umask <mask-code>`：数字形式设置权限掩码

-	选项参数
	-	`-S`：符号形式返回当前权限掩码，否则为数字形式

###	`chmod`

-	`chmod`：修改文件权限
	-	`$ chmod [ugoa][+-=][rwxst] <file>`：设置文件权限
		-	`ugoa`：*user*、*group*、*other*、*all*
		-	`+-=`：表示增加、减少、设置权限
		-	`rwxst`：读写执行权限 + 执行位标记（*set-user-id*、*set-group-id*、*sticky bit*）
	-	`$ chmod <previlege> <file>`：数字方式设置权限，`previliege` 为 4 位 8 进制数
		-	首位：执行位标记（*set-user-id*、*set-group-id*、*sticky bit*）（为 0 时常省略）
		-	后 3 位：*user*、*group*、*other* 3 种权限
	-	说明
		-	普通用户只能修改 *user* 组权限位
		-	root 用户可以修改任意用户、任意文件权限

-	选项参数
	-	`-R`：对整个目录、子文件（目录）同时修改权限

```md
$ chmod u+s file1
$ chmod 7777 file1
```

###	`ls`

-	`ls`：列出当前工作目录目录、文件信息 
	-	`$ ls <op> <file>`

-	选项参数
	-	`-a`：列出所有文件，包括隐藏文件
	-	`-l`：文件详细信息
	-	`-t`：按最后修改时间排序
	-	`-S`：按文件大小排序
	-	`-r`：反向排序
	-	`-h`：显示文件大小时增加可读性
	-	`-F`：添加描述符到条目后
		-	`@`：符号链接
		-	`*`：文件
		-	`/`：目录
	-	`-i`：显示索引节点

####	输出说明

![ls_results.png](imgs/ls_results.png)

-	文件类型
	-	`-`；普通文件
	-	`d`：目录
	-	`l`：link，符号链接
	-	`s`：socket
	-	`b`：block，块设备
	-	`c`：charactor，字符设备（流）
	-	`p`：FIFO Pipe

-	文件权限：包括 9 个字符，包含 $3 * 3 + 3$ 权限设置
	-	字符分为三组，表示不同类型用户权限
		-	第 1-3 字符：*owner* 文件属主权限
		-	第 4-6 字符：*group* 同组用户权限
		-	第 7-9 字符：*other* 其他用户权限
	-	常规权限字符含义
		-	`r`：读，每组首位
		-	`w`：写，每组中间位
		-	`x`：执行，每组末位
		-	`-`：没有相应此权限
	-	各组的执行位可能为其他字符表示特殊权限
		-	*owner* 组 `s`：文件 *set-user-id*、执行同时被置位
		-	*owner* 组 `S`：文件 *set-user-id* 被置位，执行权限未置位
		-	*group* 组 `s`：文件 *set-group-id* 、执行权限同时被置位
		-	*group* 组 `S`：文件 *set-group-id* 被置位，执行权限未置位
		-	*other* 组 `t`：文件 *sticky bit* 、执行权限均被置位
		-	*other* 组 `T`：文件 *sticky bit* 被置位，执行权限未置位

	> - 关于权限具体含义，参见`linux/kernel/file_system`
	> - 权限设置，参见`linux/shell/cmd_fds`

-	文件硬链接数量
	-	一般文件：硬链接数目
	-	目录：也即 *目录中第一级子目录个数 + 2*
-	文件属主名
-	文件属主默认用户组名
-	文件大小（Byte）
-	最后修改时间
-	文件名

###	`mktemp`

-	`mktemp`：安全创建临时文件
	-	说明
		-	不会检查临时文件是否存在，但支持唯一文件名、清除机制
		-	打印临时文件名，可通过命令扩展获取

-	选项参数
	-	`-d`：创建临时目录
	-	`-p <DIR>`：指定临时文件所在目录
	-	`-t <TMPL>`：指定临时文件名模板
		-	模板末尾至少包含三个连续 `X`，表示随机字符
		-	默认模板为 `tmp.XXXXXXXXXX`

##	命令总结

-	文件系统状态
	-	`du`：显示目录、文件磁盘占用量（文件系统数据库情况）
	-	`df`：文件系统信息 
	-	`fdisk`：查看系统分区
	-	`mkfs`：格式化分区
	-	`fsck`：检查修复文件系统
	-	`mount`：查看已挂载的文件系统、挂载分区 
	-	`umount`：卸载指定设备
	-	`free`：查看系统内存、虚拟内存占用

-	目录、文件操作
	-	`pwd`：显示当前工作目录绝对路径
	-	`cd`：更改工作目录路径 **（*Bash* 命令）**
		-	缺省回到用户目录
		-	`-`：回到上个目录
	-	`ls`：列出当前工作目录目录、文件信息
	-	`tree`：树状图逐级列出目录内容
	-	`dirs`：显示目录列表
	-	`touch`：创建空文件或更改文件时间
	-	`mkdir`：创建目录
	-	`mktemp`：创建临时文件，输出临时文件名
	-	`rmdir`：删除空目录 
	-	`cp`：复制文件和目录
	-	`mv`：移动、重命名文件、目录 
	-	`rm`：删除文件、目录
		-	删除目录符号链接时，末尾带 `/` 被视为删除目录全部内容
	-	`file`：查询文件的文件类型
	-	`dirname`：输出给出参数字符串中的目录名
		-	返回结果不包括 `/`
		-	若参数中无任何 `/`，则返回 `.`
	-	`basename`：输出给出参数字符串中的文件、目录名
	-	`ln`：创建链接文件
	-	`stat`：显示文件、文件系统状态

-	文件、目录权限、属性
	-	`chown`：更改文件、目录的用户所有者、组群所有者
	-	`chgrp`：更改文件、目录所属组
	-	`umask`：显示、设置文件、目录创建默认权限掩码
	-	`getfacl`：显示文件、目录 *ACL*
	-	`setfacl`：设置文件、目录 *ACL*
	-	`chacl`：更改文件、目录 *ACL* 
	-	`lsattr`：查看文件、目录属性
	-	`chattr`：更改文件、目录属性
	-	`umask`：设置权限掩码
	-	`chmod`：设置文件、目录权限

-	查找文件
	-	`find`：列出文件系统内符合条件的文件
	-	`whereis`：插卡指定文件、命令、手册页位置
	-	`whatis`：在whatis数据库中搜索特定命令
	-	`which`：显示可执行命令路径
	-	`type`：输出命令信息
		-	可以用于判断命令是否为内置命令



