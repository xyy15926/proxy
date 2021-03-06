---
title: 常用工具
tags:
  - Linux
  - Shell命令
categories:
  - Linux
  - Shell命令
date: 2019-07-31 21:10:52
updated: 2019-02-24 15:57:09
toc: true
mathjax: true
comments: true
description: 常用工具
---

##	获取命令系统帮助

###	`help`

重看内部shell命令帮助信息（常用）

###	`man`

显示在线帮助手册（常用）

###	`info`

info格式的帮助文档

##	打印、日期、时间

###	`echo`

输出至标准输出

###	`env`

打印当前所有环境变量

###	`cal`

显示日历信息

###	`date`

`date`：显示、设置系统日期时间

```shell
date -d <time> "+<format>"
```

-	`-d`：指定时间，缺省今天
-	`+`：指定输出格式
	-	`%Y-%m-%d %h-%M-%S`：年月日时（24时）分秒
	-	`%a/%A`：星期缩写、完整
	-	`%b/%B`：月份缩写、完整
	-	`%D`：`MM/DD/YY`
	-	`%F`：`YYYY-MM-DD`

###	`hwclock`

查看、设置硬件时钟

###	`clockdiff`

主机直接测量时钟差

###	`rdate`

通过网络获取时间

###	`sleep`

暂停指定时间


##	数值计算

###	`bc`

任意精度计算器

###	`expr`

将表达式值打印到标准输出，注意转义


##	归档、压缩

###	`tar`

多个文件保存进行归档、压缩

###	`gzip`

压缩、解压缩gzip文件

###	`gunzip`

解压缩gzip文件

###	`zcmp`

调用diff比较gzip压缩文件

###	`unzip`

解压缩zip文件

###	`zip`

压缩zip文件

###	`zcat`

查看zip压缩文件

###	`zless`

查看zip压缩文件

###	`zipinfo`

列出zip文件相关详细信息

###	`zipsplit`

拆分zip文件

###	`zipgrep`

在zip压缩文件中搜索指定字符串、模式

###	`zmore`

查看gzip/zip/compress压缩文件


###	`rpm2cpio`

将rpm包转变为cpio格式文件，然后可以用cpio解压

```
$ rpm2cpio rpm_pkg | cpio -div
```

##	远程连接服务器

###	`ssh`

`ssh`：ssh登陆服务器

```shell
ssh <user_name>@<host> [-p <port>] [-i <private_key>]
	[-F <config_file>]
```

-	参数
	-	`-F`：指定配置文件，简化命令，缺省`~/.ssh/config`
		-	有时会覆盖其余配置，可以指定`-F /dev/null`避免
	-	`-p`：指定端口
	-	`-i`：指定使用的私钥
	-	`-T`：测试连接

####	`ssh-keygen`

`ssh-keygen`：生成ssh密钥

```sh
 # 生成位长为4096的RSA密钥
$ ssh-keygen -t rsa -b 4096 -C "xyy15926@163.com"
 # 更改现有私钥密钥
$ ssh-keygen -p [-f keyfile] [-m format] [-N new_pwd] [-p old_pwd]
```

-	参数
	-	`-t [dsa|ecdsa|ecdsa-sk|ed25519|ed25519-sk|ras]`：
		生成密钥类型
	-	`-f <output_keyfile>`：密钥保存文件
	-	`-b <bits>`：密钥位数
	-	`-C <comment>`
	-	`-p`：修改现有密钥密码

-	在生成密钥时可以设置密钥，避免计算机访问权限被获取后密钥
	被用于访问其他系统
	-	设置有密钥的私钥每次使用都需要输入密码
	-	`ssh-agent`可以用于管理私钥密码，将私钥添加给
		`ssh-agent`无需每次输入密码

####	`ssh-add`

`ssh-add`：管理`ssh-agent`代理密钥

```sh
 # 缺省将`$HOME/.ssh/id_rsa`私钥添加至`ssh-agent`
$ ssh-add <keyfile>
```

-	参数
	-	`-l`：查看代理中私钥
	-	`-L`：查看代理中私钥对应公钥
	-	`-d <keyfile>`：移除指定私钥
	-	`-D`：移除所有私钥
	-	`-x/-X`：锁定/解锁`ssh-agent`（需要设置密码）
	-	`-t <seconds>`：密钥时限

####	`ssh-agent`

`ssh-agent`：控制和保存公钥验证所使用的私钥程序

-	参数
	-	`-k`：关闭当前`ssh-agent`进程

-	启动方式

	```sh
	# 创建默认子shell，在子shell中运行`ssh-agent`
	# 会自动进入子shell中，退出子shell则`ssh-agent`关闭
	$ ssh-agent $SHELL
	# 启动后台`ssh-agent`进程，需手动关闭`ssh-agent`进程
	$ eval `ssh-agent`
	```

-	说明
	-	向`ssh-agent`添加私钥需要输入私钥密码
		（无密码私钥无管理必要）
	-	`ssh-agent`不会默认启动，应是每次需要大量使用私钥
		前启动、添加私钥，然后关闭

###	`/etc/init.d/sshd`

`sshd`：ssh连接的服务器守护程序（进程）

```shell
$ sudo systemctl start sshd
	# 使用`systemctl`启动
$ /etc/init.d/sshd restart
	# 直接启动进程
```

###	`scp`

`scp`：secure cp，安全传输（cp）文件

-	本机到远程
	```shell
	$ scp /path/to/file user_name@host:/path/to/dest
	```

-	远程到本机
	```shell
	$ scp user_name@host:/path/to/file /path/to/dest
	```

	-	这个命令应该在本机上使用，不是ssh环境下
	-	ssh环境下使用命令表示在远程主机上操作 
	-	而本机host一般是未知的（不是localhost）

-	远程到远程


###	`rsync`

实现本地主机、远程主机的文本双向同步

####	同步两种模式

同步过程由两部分模式组成

-	决定哪些文件需要同步的检查模式
	-	默认情况下，`rsync`使用*quick check*算法快速检查
		源文件、目标文件大小、mtime是否一致，不一致则
		需要传输
	-	也可以通过指定选项改变检查模式

-	文件同步时的同步模式：文件确定被同步后，在同步发生
	之前需要做哪些额外工作
	-	是否删除目标主机上比源主机多的文件
	-	是否要先备份已经存在的目标文件
	-	是否追踪链接文件

####	工作方式

-	本地文件系统同步：本质是管道通信
	```shell
	$ rsync [option..] src... [dest]
	```

-	本地主机使用远程shell和远程主机通信
	```shell
	$ rsync [option...] [user@]host:src... [dest]
	$ rsync [option...] src... [user@]host:dest
	```
	-	本质上是管道通信

-	本地主机通过socket连接远程主机的rsync
	```shell
	$ rsync [option...] rsync://[user@]host[:port]/src... dest
	$ rsync [option...] [user@]host::src... [dest]
	$ rsync [option...] src... rsync://[user@]host[:port]/dest
	$ rsync [option...] src... [user@]host::dest
	```
	-	需要远主机运行rsync服务监听端口，等待客户端连接

-	通过远程shell临时派生rsync daemon，仅用于临时读取daemon
	配置文件，同步完成后守护进程结束

	-	语法同shell远程主机通信，指定`--rsh`/`-e`选项

####	参数说明

-	可以有多个源文件路径，最后一个是目标文件路径
-	如果只有一个路径参数，则类似于`ls -l`列出目录
-	注意：如果原路径是目录
	-	`/`结尾表示目录中文件，不包括目录本身
	-	不以`/`结尾包括目录本身

####	常用选项

#####	同步模式选项

-	`v`/`-vvvv`：显示详细/更详细信息
-	`p`/`--partial --progress`：显示进度
-	`-n`/`--dry-run`：测传传输
-	`-a`/`--archive`：归档模式，递归传输并保持文件属性
-	`-r`/`--recursive`：递归至目录
-	`-t`/`--times`：保持mtime属性
	-	建议任何时候都使用，否则目标文件mtime设置为系统时间
-	`-o`/`--owner`：保持属主
-	`-g`/`--group`：保持group
-	`-p`/`--perms`：保持权限（不包括特殊权限）
-	`-D`/`--device --specials`：拷贝设备文件、特殊文件
-	`-l`/`--links`：如果目标是软链接文，拷贝链接而不是文件
-	`z`：传输时压缩

-	`w`/`--whole-file`：使用全量传输
	-	网络带宽高于磁盘带宽时，此选项更高效

-	`R`/`--relative`：使用相对路径，即在目标中创建源中指定
	的相对路径

	```shell
	$ rsync -R -r /var/./log/anaconda /tmp
		# 将会在目标创建`/tmp/log/anaconda`
		# `.`是将绝对路径转换为相对路径，其后的路径才会创建
	```

-	`--delete`：以源为主，对DEST进行同步，多删、少补
-	`-b`/`--backup`：对目标上已经存在文件做备份的
	-	备份文件名使用`~`做后缀
-	`--backup-dir`：指定备份文件保存路径，不指定为同一目录
-	`--remove-source-file`：删除源中已成功传输文件

#####	连接

-	`--port`：连接daemon时端口号，默认873
-	`--password-file`：daemon模式时的密码文件
	-	可以从中读取密码文件实现非交互式
	-	是rsync模块认证密码，不是shell认证密码
-	`-e`：指定所需要的远程shell程序，默认ssh
	```shell
	$ rsync -e "ssh -p 22 -o StrictHostKeyChecking=no" /etc/fstab
	```
-	`--rsh`：使用rsync deamon进行同步

#####	检查模式选项

-	`--size-only`：只检查文件大小，忽略mtime
-	`-u`/`--update`：进源mtime比已存在文件mtime新才拷贝
-	`-d`/`--dirs`：以不递归方式拷贝目录本身（不目录中内容）
-	`--max-size`：传输最大文件大小，可以使用单位后缀
-	`--min-size`：传输的最小文件大小
-	`--exclude`：指定派出规则排除不需要传输的文件
-	`--existing`：只更新目标端已存在文件，不存在不传输
-	`--ignore-existing`：只更新在目标段不存在文件

##	网络

###	`ping`

向被测试目的主机地址发送ICMP报文并收取回应报文

-	`-c`：要求回应的次数
-	`-i`：发送ICMP报文时间间隔
-	`-R`：记录路由过程
-	`-s`：数据包大小
-	`-t`：存活数值（路由跳数限制）

###	`ifconfig`

显示、设置网络

-	`netmask`：设置网卡子网掩码
-	`up`：启动指定网卡
-	`down`：关闭指定网络设备
-	`ip`：指定网卡ip地址

###	`netstat`

显示与网络相关的状态信息：查看网络连接状态、接口配置信息、
检查路由表、取得统计信息

-	`-a`：显示网络所有连接中的scoket
-	`-c`：持续列出网络状态
-	`-i`：显示网络界面信息表单
-	`-n`：直接使用IP地址而不是主机名称
-	`-N`：显示网络硬件外围设备号连接名称
-	`-s`：显示网络工作信息统计表
-	`-t`：显示TCP传输协议连接状况

###	`route`

查看、配置Linux系统上的路由信息

###	`traceroute`

跟踪UDP路由数据报

-	`-g`：设置来源来路由网关
-	`-n`：直接使用IP地址而不是主机名称
-	`-p`：设置UDP传输协议的通信端口
-	`-s`：设置本地主机送出数据包的IP地址
-	`-w`：超时秒数（等待远程主机回报时间）

##	磁盘

###	`df`

文件系统信息

###	`fdisk`

查看系统分区

###	`mkfs`

格式化分区

###	`fsck`

检查修复文件系统

###	`mount`

查看已挂载的文件系统、挂载分区

###	`umount`

卸载指定设备

###	`free`

查看系统内存、虚拟内存占用


