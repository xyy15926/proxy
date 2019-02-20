#	文件、目录

##	目录、文件操作

###	`pwd`

显示当前工作目录绝对路径

###	`cd`

更改工作目录路径

###	`ls`

```shell
$ ls [params] expr
```

列出当前工作目录目录、文件信息

####	参数

-	`-a`：列出所有文件，包括隐藏文件
-	`-l`：文件详细信息
	-	详细信息格式、含义参见`config_file`
-	`-t`：按最后修改时间排序
-	`-S`：按文件大小排序
-	`-r`：反向排序
-	`-h`：显示文件大小时增加可读性
-	`-F`：添加描述符到条目后
	-	`@`：符号链接
	-	`*`：文件
	-	`/`：目录
-	`-i`：显示索引节点

###	`dirs`

显示目录列表

###	`touch`

创建空文件或更改文件时间

###	`mkdir`

创建目录

###	`rmdir`

删除空目录

###	`cp`

复制文件和目录

###	`mv`

移动、重命名文件、目录

###	`rm`

删除文件、目录

###	`file`

查询文件的文件类型

###	`du`

显示目录、文件磁盘占用量（文件系统数据库情况）

####	参数

-	`-a`/`--all`：显示**所有后代**各文件、文件夹大小
	-	否则默认为显示**所有后代**文件夹大小
-	`-c`/`--total`：额外显示总和
-	`-s`/`--summarize`：仅显示总和
-	`--max-depth=[num]`：显示文件夹的深度
-	`-S`/`--separate-dirs`：文件夹大小不包括子文件夹大小

-	`-b`/`--bytes`：以byte为单位
-	`-k`/`--kilobytes`：以KB为单位
-	`-m`/`--megabytes`：以MB为单位
-	`-h`：human-readable，提升可读性
-	`-H`/`--si`：同`-h`，但是单位换算以1000为进制

-	`-x`/`--one-file-system`：以最初处理的文件系统为准，忽略
	其他遇到的文件系统
-	`-L=`/`--dereference=`：显示选项中指定的符号链接的源文件
	大小
-	`-D`/`--dereference-args`：显示指定符号链接的源文件大小
-	`-X=`/`--exclude-from=[file]`：从文件中读取指定的目录、
	文件
-	`--exclude=[dir/file]`：掠过指定目录、文件
-	`-l`/`--count-links`：重复计算hard-link

###	`wc`

统计文件行数、单词数、字节数、字符数

	-	`-l, -w, -c`
###	`tree`

树状图逐级列出目录内容

###	`cksum`

显示文件CRC校验值、字节统计

###	`mk5sum`

显示、检查MD5（128bit）校验和

###	`sum`

为文件输出校验和及块计数

###	`dirname`

输出给出参数字符串中的目录名（不包括尾部`/`）
，如果参数中不带`/`输出`.`表示当前目录

###	`basename`

输出给出参数字符串中的文件、**目录**名

###	`ln`

创建链接文件

###	`stat`

显示文件、文件系统状态

##	文件、目录权限、属性

###	`chown`

更改文件、目录的用户所有者、组群所有者

###	`chgrp`

更改文件、目录所属组

###	`umask`

显示、设置文件、目录创建默认权限掩码

###	`getfacl`

显示文件、目录ACL

###	`setfacl`

设置文件、目录ACL

###	`chacl`

更改文件、目录ACL

###	`lsattr`

查看文件、目录属性

###	`chattr`

更改文件、目录属性

###	`umask`

查看/设置权限掩码，默认`0000`

```shell
$ umask
	# 数字形式返回当前权限掩码
$ umask -S
	# 符号形式返回当前权限掩码
$ umask 0003
	# 设置权限掩码为`0003`
```

> - 权限掩码参见*linux/kernel/permissions*

###	`chmod`

关于文件、目录权限参见`config_files###文件描述`

-	普通用户只能修改user权限位
-	root用户可以修改任意用户、任意文件权限

####	参数

-	`-R`：对整个目录、子文件（目录）同时修改权限

####	操作

```shell
$ chmod [ugoa][+-=][rwxst] file
$ chmod xxxx file
```

-	`ugoa`：分别表示user、group、other、all权限位
-	`+-=`：表示增加、减少、设置权限
-	`rwxst`：表示5不同的权限
	-	`S`、`T`不是一种权限，只是一种特殊的状态
	-	设置状态时`s`时，是根据相应的`x`是否有确定`s`/`S`
	-	设置状态`t`同理

-	`xxxx`：每个8进制数表示一组权限，对应二进制表示相应权限
	是否置位
	-	第1个数字：set-user-id、set-group-id、sticky bit
	-	后面3个数字分别表示user、group、other权限
	-	第1个数字位0时可以省略（常见）

####	示例

```md
$ chmod u+s file1
$ chmod 7777 file1
```

##	显示文本文件

###	`cat`

显示文本文件内容

###	`more`

分页显示文本文件

###	`less`

回卷显示文本文件内容

###	`head`

显示文件指定前若干行

###	`tail`

实现文件指定后若干行

###	`nl`

显示文件行号、内容

##	文件处理

###	`sort`

对文件中数据排序

###	`uniq`

删除文件中重复行

###	`cut`

从文件的每行中输出之一的字节、字符、字段

###	`diff`

逐行比较两个文本文件

###	`diff3`

逐行比较三个文件

###	`cmp`

按字节比较两个文件

###	`tr`

从标准输入中替换、缩减、删除字符

###	`split`

将输入文件分割成固定大小的块

###	`tee`

将标准输入复制到指定温婉

###	`awk`

模式扫描和处理语言

	#todo

###	`sed`

过滤、转换文本的流编辑器

-	sed按行处理文本数据，每次处理一行在行尾添加换行符

```shell
$ sed [-hnV] [-e<script>][-f<script-file>][infile]
```

####	参数

-	`-e<script>/--expression=<script>`：以指定script
	处理infile（默认参数）
	-	默认不带参数即为`-e`

-	`-f<script-file>/--file=<script-file>`：以指定的script
	文件处理输入文本文件
	-	文件内容为sed的动作

-	`-i`：直接修改原文件

-	`-n/--quiet`：仅显示script处理后结果

-	`-h/--help`：帮助

-	`-V/--version`：版本信息

####	动作

-	`[n]a\string`：行添加，在`n`行后添加新行`string`
-	`[n]i\string`：行插入
-	`[n]c\string`：行替换
-	`[n,m]d`：删除，删除`n-m`行
-	`[start[,end]]p`：打印数据
-	`[start[,end]]s/expr/ctt[/g]`：正则替换

####	高级语法

####	示例

```md
$ sed '2anewline' ka.file
$ sed '2a newline' ka.file
$ sed 2anewline ka.file
$ sed 2a newline ka.file
	# 在第2行添加新行`newline`

$ sed 2,$d ka.file
	# 删除2至最后行

$ sed 2s/old/new ka.file
	# 替换第2行的`old`为`new`

$ nl ka.file | sed 7,9p
	# 打印7-9行

$ sed ":a;N;s/\n//g;ta" a.txt
	# 替换换行符
```

##	查找字符串、文件

###	`grep`

查找符合条件的字符串

###	`egrep`

在每个文件或标准输入中查找模式

###	`find`

列出文件系统内符合条件的文件

###	`whereis`

插卡指定文件、命令、手册页位置

###	`whatis`

在whatis数据库中搜索特定命令

###	`which`

显示可执行命令路径

###	`type`

输出命令信息

-	可以用于判断命令是否为内置命令

##	文本编辑器

###	`vi/vim`

###	`nano`

系统自带编辑器，有时只能编辑少部分配置文件


