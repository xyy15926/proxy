---
title: Shell命令
tags:
  - Linux
  - Shell命令
categories:
  - Linux
  - Shell命令
date: 2019-07-31 21:10:52
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Shell命令
---

-	builtin command：shell程序的一部分，包含简练的Linux系统
	命令
	-	由shell程序识别并在shell内部完成运行
	-	在linux系统加载运行时shell被加载并驻留在系统内存

-	external command：linux系统中的实用程序部分
	-	功能比较强大、包含的程序量更大
	-	需要时才被调进内存
		-	实体未包含在shell中，但是其执行过程由shell程序控制，
		shell程序管理外部命令执行路径查找、加载、存放

##	系统级查看、设置

###	进程、服务管理

-	`ps`：查看当前进程瞬时快照
-	`top`：显示当前正在运行进程
-	`pgrep`：按名称、属性查找进程
-	`pidof`：根据进程名查找正在运行的进程进程号
-	`kill`：终止进程
-	`killall`：按名称终止进程
-	`pkill`：按名称、属性终止进程
-	`timeout`：在指定时间后仍然运行则终止进程
-	`wait`：等待指定进程
-	`fuser`：显示使用指定文件、socket的进程
-	`pmap`：报告进程的内存映射
-	`lsof`：列出打开的文件
-	`chkconfig`：为系统服务更新、查询运行级别信息

####	进程、作业

#####	`&`

放在命令之后，命令后台执行

```shell
$ ./pso > pso.file 2>&1 &
	# 将`pso`放在后台运行，把终端输出（包括标准错误）
		# 重定向的到文件中
```

#####	`nohup`

不挂起job，即使shell退出

```shell
$ nohup ./pso > pso.file 2>&1 &
	# 不挂起任务，输出重定向到文件
$ nohup -p PID
	# 不挂起某个进程
	```

#####	`jobs`

列出活动的作业

`-l`：返回任务编号、进程号

#####	`bg`

恢复在后台暂停工作的作业

```shell
$ bg %n
	# 将编号为`n`的任务转后台运行
```

#####	`fg`

将程序、命令放在前台执行

```shell
$ fg %n
	# 将编号为`n`的任务转前台运行
```

#####	`setsid`

在一个新的会话中运行程序

```shell
$ setsid ./test.sh &`
	# 新会话中非中断执行程序，此时当前shell退出不会终止job
$ (./test.sh &)
	# 同`setsid`，用`()`括起，进程在subshell中执行
```

#####	`disown

```shell
$ disown -h %job_id
	# *放逐*已经在后台运行的job，
	# 则即使当前shell退出，job也不会结束
```

#####	`screen`

创建断开模式的虚拟终端

```bash
$ screen -dmS screen_test
	# 创建断开（守护进程）模式的虚拟终端screen_test
$ screen -list
	# 列出虚拟终端
$ screen -r screen_test
	# 重新连接screen_test，此时执行的任何命令都能达到nohup
	```

#####	其他

-	`<c-z>`：挂起当前任务
-	`<c-c>`：结束当前任务

####	System

-	`systemctl`：替代service更加强大

	-	`start`：启动服务
	-	`stop`：关闭服务
	-	`enable`：开机自启动服务
	-	`disable`：关闭开机自启动服务，`enable`启动后的服务
		仅仅`disable`不会立刻停止服务

	`systemctl`通过`d-bus`和systemd交流，在docker和wsl中可能
	没有systemd-daemon，此命令可能不能使用，使用`service`
	代替

-	`service`：控制系统服务

###	性能监控

-	`scar`：收集、报告、保存系统活动信息
-	`iostat`：报告CUP统计数据，设备、分区输入/输出信息
-	`iotop`：I/O监控
-	`mpstat`：报告CPU相关统计数据
-	`vmstat`：报告虚拟内存统计
-	`tload`：加载显示系统平均负载、指定tty终端平均负载
-	`time`：显示资源资源使用时间
-	`uptime`：显示系统已运行时间
-	`ipcs`：提供IPC设施信息
-	`ipcrm`：删除消息队列、信号量集、共享内存ID
-	`lslk`：列出本地锁

###	任务计划

-	`atq`：列出用户等待执行的作业
-	`atrm`：删除用户等待执行的作业
-	`watch`：定期执行程序

####	`at`

设置在某个时间执行任务

```shell
at now+2minutes/ now+5hours
	# 从现在开始计时
at 5:30pm/ 17:30 [today]
	# 当前时间
at 17:30 7.11.2018/ 17:30 11/7/2018/ 17:30 Nov 7
	# 指定日期时间
	# 输入命令之后，`<c-d>`退出编辑，任务保存
```

####	`crontab`

针对用户维护的`/var/spool/cron/crontabs/user_name`文件，其中
保存了cron调度的内容（根据用户名标识）

-	`-e`：编辑
-	`-l`：显示
-	`-r`：删除

任务计划格式见文件部分

###	字体

-	`fc-list :lang=zh`：查找系统已安装字体，参数表示中文字体
-	`fc-cache`：创建字体信息缓存文件
-	`mkfontdir/mkfontscale`：创建字体文件index
	-	还不清楚这个字体index有啥用
	-	`man`这两个命令作用差不多，只是后者是说创建scalable
		字体文件index
	-	网上信息是说这两个命令配合`fc-cache`用于安装字体，
		但是好像没啥用，把字体文件复制到`/usr/share/fonts`
		（系统）或`~/.local/share/fonts`（用户），不用执行
		这两个命令，直接`fc-cache`就行

###	环境变量

-	`export`：显示、设置环境变量，使其可在shell子系统中使用
	-	设置环境变量直接`$ ENV=value`即可，但是此环境变量
		不能在子shell中使用，只有`$ export ENV`导出后才可
	-	`-f`：变量名称为函数名称
	-	`-n`：取消导出变量，原shell仍可用
	-	`-p`：列出所有shell赋予的环境变量

###	包管理

####	`apt`

-	`install`
-	`update`
-	`remove`
-	`autoremove`
-	`clean`

####	`rpm`

####	`yum`

####	`dpkg`

####	`zypper`

####	`pacman`

##	文件、目录

###	目录、文件操作

-	`pwd`：显示当前工作目录绝对路径
-	`cd`：更改工作目录路径
-	`ls`：列出当前工作目录目录、文件信息
-	`dirs`：显示目录列表
-	`touch`：创建空文件或更改文件时间
-	`mkdir`：创建目录
-	`rmdir`：删除空目录
-	`cp`：复制文件和目录
-	`mv`：移动、重命名文件、目录
-	`rm`：删除文件、目录
-	`file`：查询文件的文件类型
-	`du`：显示目录、文件磁盘占用量（文件系统数据库情况）
-	`wc`：统计文件行数、单词数、字节数、字符数
	-	`-l, -w, -c`
-	`tree`：树状图逐级列出目录内容
-	`cksum`：显示文件CRC校验值、字节统计
-	`mk5sum`：显示、检查MD5（128bit）校验和
-	`sum`：为文件输出校验和及块计数
-	`dirname`：输出给出参数字符串中的目录名（不包括尾部`/`）
	，如果参数中不带`/`输出`.`表示当前目录
-	`basename`：输出给出参数字符串中的文件、**目录**名
-	`ln`：创建链接文件
-	`stat`：显示文件、文件系统状态

###	文件、目录权限、属性

-	`chown`：更改文件、目录的用户所有者、组群所有者
-	`chgrp`：更改文件、目录所属组
-	`umask`：显示、设置文件、目录创建默认权限掩码
-	`getfacl`：显示文件、目录ACL
-	`setfacl`：设置文件、目录ACL
-	`chacl`：更改文件、目录ACL
-	`lsattr`：查看文件、目录属性
-	`chattr`：更改文件、目录属性

####	`chmod`

-	权限分类
	-	读取权限：浏览文件/目录权限
	-	写入权限：修改文件/添加、删除、重命名目录文件内容
	-	执行权限：执行文件/进入目录

-	用户分类
	-	Owner：建立文件、目录的用户
	-	Group：文件所属群组的所有用户
	-	Other：其他用户

-	`drwxrwxrwx`表示权限
	-	1：目录权限
	-	2-4：文件所有者权限
	-	5-7：文件所属组权限
	-	8-10：其他用户权限

```md
$ chmod a+x file1
	# `a`所有用户、`o`其他用户
	# `+`添加权限、`-`删除权限
$ chmod 777 file1
	# `777`表示权限，位表示
```

###	显示文本文件

-	`cat`：显示文本文件内容
-	`more`：分页显示文本文件
-	`less`：回卷显示文本文件内容
-	`head`：显示文件指定前若干行
-	`tail`：实现文件指定后若干行
-	`nl`：显示文件行号、内容

###	文件处理

-	`sort`：对文件中数据排序
-	`uniq`：删除文件中重复行
-	`cut`：从文件的每行中输出之一的字节、字符、字段
-	`diff`：逐行比较两个文本文件
-	`diff3`：逐行比较三个文件
-	`cmp`：按字节比较两个文件
-	`tr`：从标准输入中替换、缩减、删除字符
-	`split`：将输入文件分割成固定大小的块
-	`tee`：将标准输入复制到指定温婉
-	`awk`：模式扫描和处理语言
	#todo

####	`sed`

过滤、转换文本的流编辑器

-	sed按行处理文本数据，每次处理一行在行尾添加换行符

```shell
$ sed [-hnV] [-e<script>][-f<script-file>][infile]
```

#####	参数

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

#####	动作

-	`[n]a\string`：行添加，在`n`行后添加新行`string`
-	`[n]i\string`：行插入
-	`[n]c\string`：行替换
-	`[n,m]d`：删除，删除`n-m`行
-	`[start[,end]]p`：打印数据
-	`[start[,end]]s/expr/ctt[/g]`：正则替换

#####	高级语法

#####	示例

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

###	查找字符串、文件

-	`grep`：查找符合条件的字符串
-	`egrep`：在每个文件或标准输入中查找模式
-	`find`：列出文件系统内符合条件的文件
-	`whereis`：插卡指定文件、命令、手册页位置
-	`whatis`：在whatis数据库中搜索特定命令
-	`which`：显示可执行命令路径
-	`type`：输出命令信息
	-	可以用于判断命令是否为内置命令

###	文本编辑器

-	`vi`
-	`nano`：系统自带编辑器，有时只能编辑少部分配置文件

##	常用工具

###	获取命令系统帮助

-	`help`：重看内部shell命令帮助信息（常用）
-	`man`：显示在线帮助手册（常用）
-	`info`：info格式的帮助文档
5.	`ldd`：列出二进制文件共享库依赖

###	日期、时间

-	`cal`：显示日历信息
-	`date`：显示、设置系统日期时间
-	`hwclock`：查看、设置硬件时钟
-	`clockdiff`：主机直接测量时钟差
-	`rdate`：通过网络获取时间
-	`sleep`：暂停指定时间

###	数值计算

-	`bc`：任意精度计算器
-	`expr`：将表达式值打印到标准输出，注意转义

###	归档、压缩

-	`tar`：多个文件保存进行归档、压缩
-	`gzip`：压缩、解压缩gzip文件
-	`gunzip`：解压缩gzip文件
-	`zcmp`：调用diff比较gzip压缩文件
-	`unzip`：解压缩zip文件
-	`zip`：压缩zip文件
-	`zcat`：查看zip压缩文件
-	`zless`：查看zip压缩文件
-	`zipinfo`：列出zip文件相关详细信息
-	`zipsplit`：拆分zip文件
-	`zipgrep`：在zip压缩文件中搜索指定字符串、模式
-	`zmore`：查看gzip/zip/compress压缩文件

-	`rpm2cpio`：将rpm包转变为cpio格式文件，然后可以用cpio解压
	`rpm2cpio rpm_pkg | cpio -div`

###	远程连接服务器

####	`ssh`

ssh登陆服务器

-	`$ ssh user_name@host`
-	可以使用`~/.ssh/config`配置文件简化登陆

####	`scp`

secure cp，安全传输（cp）文件

-	本机到远程
	`$ scp /path/to/file user_name@host:/path/to/dest`
-	远程到本机
	`$ scp user_name@host:/path/to/file /path/to/dest`
-	这个命令应该在本机上使用，不是ssh环境下
	-	ssh环境下使用命令表示在远程主机上操作 
	-	而本机host一般是未知的（不是localhost）

####	`ssh-keygen`

生成ssh需要的rsa密钥

-	`$ ssh-keygen -t rsa`：生成rsa密钥
-	`$ ssh-keygen -r rsa -P '' -f ~/.ssh/id_rsa`

####	`ssh-add`

添加密钥给`ssh-agent`（避免每次输入密码？）

####	`/etc/init.d/sshd`

ssh连接的服务器守护程序（进程）

```shell
$ sudo systemctl start sshd
	# 使用`systemctl`启动
$ /etc/init.d/sshd restart
	# 直接启动进程
```

###	网络

####	`ping`

向被测试目的主机地址发送ICMP报文并收取回应报文

-	`-c`：要求回应的次数
-	`-i`：发送ICMP报文时间间隔
-	`-R`：记录路由过程
-	`-s`：数据包大小
-	`-t`：存活数值（路由跳数限制）

####	`ifconfig`

显示、设置网络

-	`netmask`：设置网卡子网掩码
-	`up`：启动指定网卡
-	`down`：关闭指定网络设备
-	`ip`：指定网卡ip地址

####	`netstat`

显示与网络相关的状态信息：查看网络连接状态、接口配置信息、
检查路由表、取得统计信息

-	`-a`：显示网络所有连接中的scoket
-	`-c`：持续列出网络状态
-	`-i`：显示网络界面信息表单
-	`-n`：直接使用IP地址而不是主机名称
-	`-N`：显示网络硬件外围设备号连接名称
-	`-s`：显示网络工作信息统计表
-	`-t`：显示TCP传输协议连接状况

####	`route`

查看、配置Linux系统上的路由信息

####	`traceroute`

跟踪UDP路由数据报

-	`-g`：设置来源来路由网关
-	`-n`：直接使用IP地址而不是主机名称
-	`-p`：设置UDP传输协议的通信端口
-	`-s`：设置本地主机送出数据包的IP地址
-	`-w`：超时秒数（等待远程主机回报时间）



##	用户、登陆

-	用户类型
	-	超级用户：root用户
	-	系统用户：与系统服务相关的用户，安装软件包时创建
	-	普通用户：root用户创建，权限有限

###	显示登陆用户

-	`w`：详细查询已登录当前计算机用户
-	`who`：显示已登录当前计算机用户简单信息
-	`logname`：显示当前用户登陆名称
-	`users`：用单独一行显示当前登陆用户
-	`last`：显示近期用户登陆情况
-	`lasttb`：列出登陆系统失败用户信息
-	`lastlog`：查看用户上次登陆信息

###	用户、用户组

-	`newusers`：更新、批量创建新用户
-	`lnewusers`：从标准输入中读取数据创建用户
-	`userdel`：删除用户账户
-	`groupdel`：删除用户组
-	`passwd`：设置、修改用户密码
-	`chpassws`：成批修改用户口令
-	`change`：更改用户密码到期信息
-	`chsh`：更改用户账户shell类型
-	`pwck`：校验`/etc/passwd`和`/etc/shadow`文件是否合法、
	完整
-	`grpck`：验证用户组文件`/etc/grous/`和`/etc/gshadow`
	完整性
-	`newgrp`：将用户账户以另一个组群身份进行登陆
-	`finger`：用户信息查找
-	`groups`：显示指定用户的组群成员身份
-	`id`：显示用户uid及用户所属组群gid
-	`su`：切换值其他用户账户登陆
-	`sudo`：以superuser用户执行命令

####	`useradd/adduser`

创建用户账户（`adduser`：`useradd`命令的符号链接）

-	`-c`：用户描述
-	`-d`：用户起始目录
-	`-g`：指定用户所属组
-	`-n`：取消建立以用户为名称的群组
-	`-u`：指定用户ID

####	`usermod`

修改用户

-	`-e`：账户有效期限
-	`-f`：用户密码过期后关闭账号天数
-	`-g`：用户所属组
-	`-G`：用户所属附加组
-	`-l`：用户账号名称
-	`-L`：锁定用户账号密码
-	`-U`：解除密码锁定

####	`groupadd`

新建群组

-	`-g`：强制把某个ID分配给已经存在的用户组，必须唯一、非负
-	`-p`：用户组密码
-	`-r`：创建系统用户组

####	`groupmod`

-	`-g`：设置GID
-	`-o`：允许多个用户组使用同一个GID
-	`-n`：设置用户组名

###	登陆、退出、关机、重启

-	`login`：登陆系统
-	`logout`：退出shell
-	`exit`：退出shell（常用）
-	`rlogin`：远程登陆服务器
-	`poweroff`：关闭系统，并将关闭记录写入`/var/log/wtmp`
	日志文件
-	`ctrlaltdel`：强制或安全重启服务器
-	`shutdown`：关闭系统
-	`halt`：关闭系统
-	`reboot`：重启系统
-	`init 0/6`：关机/重启

##	硬件

###	磁盘

-	`df`：文件系统信息
-	`fdisk`：查看系统分区
-	`mkfs`：格式化分区
-	`fsck`：检查修复文件系统
-	`mount`：查看已挂载的文件系统、挂载分区
-	`umount`：卸载指定设备

##	服务

###	常用服务

-	`mysqld`/`mariadb`：
-	`sshd`
-	`firewalld`
