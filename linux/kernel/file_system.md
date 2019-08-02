---
title: Linux文件系统设计
tags:
  - Linux
categories:
  - Linux
date: 2019-07-31 21:10:51
updated: 2019-02-19 17:19:50
toc: true
mathjax: true
comments: true
description: Linux文件系统设计
---

##	文件、文件系统

###	文件

-	文件：被命名的存储在某种介质上的一组信息的集合
	-	用于存储信息的基本结构
	-	技术上，文件时一指向相应*inode*（索引节点）

-	目录（文件）：用于管理、组织大量文件

> - *inode*：索引节点，包含文件真正的信息，可用`$ ls -i`查看
> - 文件类型表示见*config_file*

###	文件系统

包含在存储设备（分区）的目录结构、文件组织方法

-	基于被划分的存储设备上的逻辑单位，一种定义文件命名、存储
	、组织、取出的方法
-	涉及：目录树、分区上文件的排列
-	存储设备：磁盘、光盘、网络存储、虚拟数据
-	存储设备可以包含多个文件系统
-	进入文件系统首先需要挂载文件系统

###	文件系统分类

|FileSystem|File Size Limit|Filesystem Size Limit|
|-----|-----|-----|
|ext2/ext3 with 1KB blocksize|16448MB|2048GB|
|ext2/ext3 with 2KB blocksize|256GB|8192GB|
|ext2/ext3 with 4KB blocksize|2048GB|8192GB|
|ext2/ext3 with 8KB blocksize|65568GB|32TB|
|ReiserFS3.5|2GB|16TB|
|ReiserFS3.6|1EB|16TB|
|XFS|8EB|8EB|
|JFS with 512B blocksize|8EB|512TB|
|JFS with 4KB blocksize|8EB|4PB|
|NFSv2(client side)|2GB|8EB|
|NFSv3(client side)|8EB|8EB|

####	EXT2

-	Linux的正宗文件系统，早期常用
-	支持*undelete*，误删文件可恢复，但是操作比较麻烦

####	EXT3

-	由EXT2发展而来
-	支持大文件
-	不支持反删除，Redhat、Fedora推荐

####	ReiserFS

-	支持大文件
-	支持反删除，操作简单

##	权限设计

-	`r`：读文件
-	`w`：修改、删除文件
-	`x`：可以执行文件
-	`s`：强制位权限（固化用户/组权限）
	-	`set-user-id`：user执行权限位出现
	-	`set-group-id`：group执行权限位出现
-	`t`：粘滞位权限（在swap中停留）

###	权限判断规则

-	linux中权限是根据`user-id`、`group-id`判断用户和资源
	关系，然后选择相应的类型用户（user、group、other）权限组
	判断是否有相应权限

-	需要注意的是，访问资源实际上不是用户，而是用户开启的
	**进程**，所以这里涉及了4中不同的**用户标识**

	-	`real-user-id`：UID，用户id

	-	`real-group-id`：GID，用户默认组id

	-	`effective-user-id`：是针对进程（可执行文件）而言，
		指内核真正用于判断进程权限的**user-id**

	-	`effective-group-id`：同`effective-user-id`，内核
		判断真正判断进程权限的**group-id**

-	一般情况下`effective-user-id`就是`read-user-id`，即启动
	进程用户的UID，所以一般来说用户创建的进程的对资源访问
	权限就是就是自身权限

###	可执行文件权限

-	`r`：读文件
-	`w`：写文件
-	`x`：执行文件

####	`s`权限

当可执行文件具有`set-user-id`权限时

-	其他用户执行该文件启动的进程的`effective-user-id`不再是
	`real-user-id`，即和执行用户的UID不再一致，而是用户属主
	的UID

-	内核根据进程`effective-user-id`判断进程权限，进程的权限
	实际上同属主的权限，而不是执行用户权限

-	这样用户就在执行这种可执行文件，暂时拥有该可执行文件属主
	执行该可执行文件权限，否则可能由于进程访问其他资源原因
	无法正常执行

-	可看作是将属主的部分权限（在该文件上涉及到的权限）
	**固化**在文件上

`set-group-id`类似的可以看作是将属主默认组权限固化在文件上

####	`t`权限

-	文件被执行时，文本段会被加载到swap中，程序结束后仍然
	保留在swap中

-	下次执行文件时，文本段直接从swap中加载，因为swap为连续
	block，加载速度会提高


###	目录权限说明

linux中目录是一种特殊的文件，其包含目录下所有文件（包括
子目录）的文件名、i-node号

-	`r`：列出目录下所有文件

-	`w`：增加、删除、重命名目录下的文件

-	`x`：可以是（搜索）路径的一部分

	-	即必须对要访问的文件中路径中所有目录都有执行权限
	-	可以将目录执行权限看作是**过境证明**

-	`s`：好像没啥用

-	`t`：用户只能增加、删除、重命名目录下**属于**自己文件

	-	对`w`权限补充，否则用户拥有目录`w`权限则可以操作目录
		下所有文件
	-	`/home`目录权限就是1777，设置有粘滞位权限

###	权限掩码

文件/目录默认权限 = 现有权限（`0777`）**减去**权限掩码

> - 权限掩码设置参见*linux/shell/cmd_fs*、
	*linux/shell/config_file*、

