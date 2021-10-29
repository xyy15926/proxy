---
title: Linux 文件系统配置
categories:
  - Linux
  - File System
tags:
  - Linux
  - Kernel
  - File System
  - Ext2/3
  - Ext4
  - Inode
  - Block
  - Block Group
date: 2021-07-29 16:31:00
updated: 2021-07-29 16:33:46
toc: true
mathjax: true
description: 
---

##	硬件

###	磁盘挂载

####	`/etc/fstab`

`/etc/fstab`：包含存储设备、文件系统信息，配置**自动挂载**
各种文件系统格式硬盘、分区、可移动设备、远程设备
（即`mount`参数存盘）

```cnf
<fs> <mountpoint> <type> <opts> <dump> <pass>
```

-	`<fs>`：挂载设备/分区名
	-	`/dev/sda`：设备/分区名
	-	`UUID=xxxxx`：使用设备UUID值表示设备
	-	`tmpfs`：tmpfs分区，默认被设置为内存的一半（可在
		`<opts>`中添加`size=2G`指定最大空间）

	> - 所有设备/分区都有唯一UUID，由文件系统生成工具`mkfs.`
		创建文件系统时生成

-	`<mountpoint>`：挂载点，路径名（文件夹）
	-	`/`
	-	`/boot`

	> - 路径名中包含可以空格使用`\040`（8进制）表示

-	`<type>`：文件系统类型
	-	`ext2`
	-	`ext3`
	-	`reiserfs`
	-	`xfs`
	-	`jfs`
	-	`iso9660`
	-	`vfat`
	-	`ntfs`
	-	`swap`
	-	`tmpfs`：临时文件系统，驻留在交换分区、内存中
		-	提高文件访问速度，保证重启时自动清除这些文件
		-	常用tmpfs的目录：`/tmp`、`/var/lock`、`/var/run`
	-	`auto`：由`mount`自动判断

-	`<opts>`：文件系统参数

	-	`noatime`：关闭atime特性
		-	不更新文件系统上inode访问记录，提高性能，否则
			即使从缓冲读取也会产生磁盘写操作
		-	老特性可以放心关闭，能减少*loadcycle*
		-	包含`nodiratime`

	-	`nodiratime`：不更新文件系统上目录inode访问记录

	-	`relatime`：实时更新inode访问记录，只有记录中访问
		时间早于当前访问才会被更新
		-	类似`noatime`，但不会打断其他程序探测，文件在
			上次访问后是否需被修改（的进程）

	-	`auto`：在启动、终端中输入`$ mount -a`时自动挂载
	-	`noauto`：手动挂载

	-	`ro`：挂载为自读权限
	-	`rw`：挂载为读写权限

	-	`exec`：设备/分区中文件**可执行**
	-	`noexec`：文件不可以执行

	-	`sync`：所有I/O将以同步方式进行
	-	`async`：所有I/O将以异步方式进行

	-	`user`：允许任何用户挂载设备，默认包含
		`noexec,nosuid,nodev`（可被覆盖）
	-	`nouser`：只允许root用户挂载

	-	`suid`：允许*set-user/group-id*（固化权限）执行
		> - `set-user/group-id`参见`linux/shell/config_files`
	-	`nosuid`：不允许*set-user/group-id*权限位

	-	`dev`：解释文件系统上的块特殊设备
	-	`nodev`：不解析文件系统上块特殊设备

	-	`umask`：设备/分区中**文件/目录**默认权限掩码
		> - 权限掩码参见`linux/kernel/file_system.md`
	-	`dmask`：设备/分区中**目录**默认权限掩码
	-	`fmask`：设备/分区中**普通文件**默认权限掩码

	-	`nofail`：设备不存在则直接忽略不报错
		-	常用于配置外部设备

	-	`defaults`：默认配置，等价于
		`rw,suid,exec,auto,nouser,async`

-	`<dump>`：决定是否dump备份
	-	`1`：dump对此文件系统做备份
	-	`0`：dump忽略此文件系统
	
	> - 大部分用户没有安装dump，应该置0

-	`<pass>`：是否以fsck检查扇区，按数字递增依次检查（相同
	则同时检查）
	-	`0`：不检验（如：swap分区、`/proc`文件系统）
	-	`1`：最先检验（一般根目录分区配置为`1`）
	-	`2`：在1之后检验（其他分区配置为`2`）

> - `/etc/fstab`是启动时配置文件，实际文件系统挂载是记录到
	`/etc/mtab`、`/proc/mounts`两个文件中

> - 根目录`/`必须挂载，必须先于其他的挂载点挂载

##	文件系统配置

###	*Ext* 配置文件

> - `/etc/mke2fs.conf`

```cnf
[defaults]
	base_features = sparse_super,large_file,filetype,resize_inode,dir_index,ext_attr
	default_mntopts = acl,user_xattr
	enable_periodic_fsck = 0
	blocksize = 4096				# 块大小
	inode_size = 256				# Inode 大小
	inode_ratio = 16384				# 分配 Inode 号间隔

[fs_types]
	ext3 = {
		features = has_journal
	}
	ext4 = {
		features = has_journal,extent,huge_file,flex_bg,metadata_csum,64bit,dir_nlink,extra_isize
		inode_size = 256
	}

[options]
	fname_encoding = utf8
```

