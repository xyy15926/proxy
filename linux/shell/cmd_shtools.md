#	常用工具

##	获取命令系统帮助

###	`help`

重看内部shell命令帮助信息（常用）

###	`man`

显示在线帮助手册（常用）

###	`info`

info格式的帮助文档

###	`ldd`

列出二进制文件共享库依赖


##	日期、时间

###	`cal`

显示日历信息

###	`date`

显示、设置系统日期时间

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

ssh登陆服务器

-	`$ ssh user_name@host`
-	可以使用`~/.ssh/config`配置文件简化登陆

###	`scp`

secure cp，安全传输（cp）文件

-	本机到远程
	`$ scp /path/to/file user_name@host:/path/to/dest`
-	远程到本机
	`$ scp user_name@host:/path/to/file /path/to/dest`
-	这个命令应该在本机上使用，不是ssh环境下
	-	ssh环境下使用命令表示在远程主机上操作 
	-	而本机host一般是未知的（不是localhost）

###	`ssh-keygen`

生成ssh需要的rsa密钥

-	`$ ssh-keygen -t rsa`：生成rsa密钥
-	`$ ssh-keygen -r rsa -P '' -f ~/.ssh/id_rsa`

###	`ssh-add`

添加密钥给`ssh-agent`（避免每次输入密码？）

###	`/etc/init.d/sshd`

ssh连接的服务器守护程序（进程）

```shell
$ sudo systemctl start sshd
	# 使用`systemctl`启动
$ /etc/init.d/sshd restart
	# 直接启动进程
```

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


