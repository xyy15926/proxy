---
title: Linux 远程工具
categories:
  - Linux
  - Tool
tags:
  - Linux
  - Tool
  - Remote
  - Rsync
  - SSH
date: 2021-07-29 21:32:17
updated: 2021-07-29 21:42:23
toc: true
mathjax: true
description: 
---

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


###	SSH

####	`/etc/ssh/sshd_config`

全局系统ssh配置

```shell
RSAAuthentication
	# 去掉注释启用RSA认证
PubkeyAuthentication yes
	# 启用公私钥配对认证方式
AuthorizedKeyFile .ssh/authorized_keys
	# 设置被认证的公钥存放文件
	# 默认为`~/.ssh/authorized_keys`中
	# 将其他主机公钥写入其中，即可从其使用密钥认证免密访问
```

####	`~/.ssh`

#####	`authorized_keys`

在`/etc/.ssh/sshd_config`中默认的存储其他主机公钥的文件，

-	使用在其中的公钥可以使用公钥登陆，无需密码
-	必须设置只有root用户有更改权限有效，否则报错

```shell
---pubkey---
```

#####	`id_rsa.pub`

生成公钥

#####	`config`

ssh访问远程主机配置

-	每条配置对应一个ssh连接，配置后可以使用`link_name`直接
	连接

```conf
Host <link_name>
	HostName <host>
	User <user_name>
	Port <port>
	IdentityFile <private_key>
	IdentitiesOnly yes
	PreferredAuthentications publickey
```

-	默认私钥文件为：`~/.ssh/id_rsa`，可以指定特定私钥

> - 此文件中配置是ssh协议参数封装，适合任何可以使用ssh协议
	场合：`ssh`、`git`、`scp`

