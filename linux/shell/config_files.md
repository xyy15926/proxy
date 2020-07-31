---
title: Linux特殊文件
tags:
  - Linux
  - Shell命令
categories:
  - Linux
  - Shell命令
date: 2019-07-31 21:10:52
updated: 2019-06-26 01:40:44
toc: true
mathjax: true
comments: true
description: Linux特殊文件
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

##	任务、作业

###	Crontab文件

####	`/var/spool/cron/crontabs/user_name`

使用用户名标识的每个用户的定时任务计划文件，文件中每行为一个
cron调度内容（也可以使用`crontab -e`直接编辑）

-	共6个字段，前5个字段为执行调度时间分配
	-	分钟、小时（24）、日期（31）、月份（12）、星期
	-	`*`表示每个当前时间
	-	`-`时间区间
	-	`,`时间分隔
	-	`[range]/[n]`：范围内每隔多久执行一次，范围为`*`可省

-	最后字段为调度内容

-	例

	```shell
	3,15 8-11 */2 * * command
		# 每隔两天的8-11点第3、15分执行
	3,15 8-11 * * 1 command
		# 每个星期一的上午8-11点的第3、15分钟执行
	```

####	`/etc/crontabs`

系统级crontab文件，类似于用户crontab文件，只是多一个用户名
字段位于第6个

##	系统设置`/etc/sysconfig`

###	`/etc/sysconfig/network[-scripts]`

文件夹包含网卡配置文件

####	一般linux

-	`ifcfg-ethXX`：linux默认ethernet网卡配置文件名称
-	`ifcfg-wlanXX`：无线局域网网卡配置文件名称

####	CentOS7网卡名称

-	前两个字符
	-	`en`：Enthernet以太网
	-	`wl`：WLAN无线局域网
	-	`ww`：WWAN无线广域网

-	第3个字符
	-	`o<index>`：on-board device index number, 
	-	`s<slot>`：hotplug slot index number
	-	`x<MAC>`：MAC address
	-	`p<bus>s<slot>`：PCI geographical location/USB port
		number chain

-	命名优先级
	-	板载设备：固件、BIOS提供的索引号信息可读：`eno1`
	-	固件、BIOS提供的PCI-E热拔插索引号可读：`ens33`
	-	硬件接口物理位置：`enp2s0`
	-	linux传统方案：`eth0`
	-	接口MAC地址：`enxXXXXXXXXXXXXXXXXX`，默认不使用，
		除非用户指定使用

-	示例
	-	`enoXX`：主板bios内置网卡
	-	`ensXX`：主板bios内置PCI-E网卡
	-	`enpXXs0`：PCI-E独立网卡

#####	恢复传统命名方式

编辑grub文件，然后使用`grub2-mkconfig`重新生成
`/boot/grub2/grub.cfg`，这样系统就会根据传统linux网卡文件
命名方式查找配置文件

```cnf
	# `/etc/sysconfig/grub`
GRUB_CMDLINE_LINUX="net.ifnames=0 biosdevname=0"
	# 为`GRUB_CMDLINE_LINUX`增加2个参数
```

####	CentOS7配置格式

```cnf
DEVICE=ens33				# 网络连接名称（实际显示的网络名称）
NAME=ens33					# 网卡物理设备名称
TYPE = Ethernet				# 网卡类型：以太网
ONBOOT=yes					# 开机启动：是
DEFROUTE=yes				# 是否设置此网卡为默认路由：是
NM_CONTROLLED=yes			# 是否可由Network Manager托管：是

BOOTPROTO=dhcp				# 网卡引导协议
							# `none`：禁止DHCP
							# `static`：启用静态IP地址
							# `dhcp`：开启完整DHCP服务
IPADDR=xxx.xxx.xxx.xxx		# IP地址
IPV4_FAILURE_FATAL=no		# IPV4致命错误检测（失败禁用设备）：否
IPV6INIT=yes				# IPV6自动初始化：是
IPV6_AUTOCONF=yes			# IPV6自动配置：是
IPV6_DEFROUTE=yes			# IPV6是否可为默认路由：是
IPV6_FAILURE_FATAL=no		# IPV6致命错误检测（失败禁用设备）：否
IPV6_ADDR_GEN_MODE=stable-privacy
							# IPV6地地址生成模型：stable-privacy

DNS1=xxx.xxx.xxx.xxx		# DNS服务器地址1
DNS2=xxx.xxx.xxx.xxx		# DNS服务器地址2
PEERDNS=no					# 是否允许DHCP获得DNS覆盖本地DNS：否
GATEWAY=xxx.xxx.xxx.xxx		# 网关
PREFIX=24					# 子网掩码使用24位
NETMASK=255.255.255.0		# 子网掩码
							# 两个参数应该只需要使用一个即可
BROADCAST=	

HWADDR=xxxxxxxxxx			# 接口MAC地址
							# 配置文件都会被执行，`HWADDR`
							# 能匹配上硬件，配置才会生效，
							# 否则硬件使用默认配置
							# 若多个配置文件配置相同`HWADDR`
							# 则操作network服务时报错
UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
							# 通用唯一识别码：不能相同（注意虚拟机）

PROXY_METHOD=none			# 代理方式：无
BROWSER_ONLY=no				# 仅浏览器：否

USERCTL=no					# 是否允许非root用户控制此设备：否
MASTER=						# 主设备名称？
SLAVE=						# 从社会名称？
NETWORK=					# 网络地址
```

##	系统引导

###	Grub

####	`/etc/sysconfig/grub`

GRUB配置文件，实际上是`/etc/default/grub`的软连接

##	服务

###	系统服务

-	服务`systemctl`脚本目录优先级从低到高

	-	`[/usr]/lib/systemd/system`：系统、应用默认存放位置
		，随系统、应用更新会改变

	-	`/etc/systemd/system`：**推荐在此自定义配置**，避免
		被更新覆盖

		```cnf
		.include /lib/systemd/system/<service_name>.service
		# customized changes
		```

	-	`/run/systemd/system`：进程运行时创动态创建服务文件
		目录，仅修改程序运行时参数才会修改

-	自定义系统服务方法

	-	创建`<service_name>.service`文件，其中写入配置
	-	创建`<service_name>.service.d`目录，其中新建`.conf`
		文件写入配置

> - 系统服务控制命令`systemctl`参见*linux/shell/cmd_sysctl*

####	`.service`

`<some_name>.service`：服务配置unit文件

```cnf
[Unit]					# 包含对服务的说明
Description=			# 服务简单描述
Documentation=			# 服务文档
Requires=				# 依赖unit，未启动则当前unit启动失败
Wants=					# 配合unit，不影响当前unit启动
BindsTo=				# 类似`Requires`，若其退出则当前unit
							# 退出
Before=					# 若该字段指定unit要启动，则必须在
							# 当前unit之后启动
After=					# 若该字段指定unit要启动，则必须在
							# 当前unit之前启动
Conflicts=				# 冲突unit
Condition=				# 条件，否则无法启动
Assert=					# 必须条件，否则启动失败


[Service]				# 服务具体运行参数设置，仅
							# *service unit*有该字段
Type=					# 服务启动类型
PIDFile=				# 存放PID文件路径
RemainAfterExit=		# 进程退出后是否认为服务仍处于激活

ExecStartPre=			# 服务启动前执行的命令
ExecStartPost=			# 服务启动后执行的命令
ExecStart=				# 服务具体执行、重启、停止命令
ExecReload=
ExecStop=

Restart=				# 重启当前服务的情况
RestartSec=				# 自动重启当前服务间隔秒数
TimeoutSec=				# 停止当前服务之前等待秒数

PrivateTmp=				# 是否给服务分配临时空间
KillSignal=SIGTERM
KillMode=mixed

Environmen=				# 指定环境变量

[Install]
WantedBy=				# unit所属的target，`enable`至相应
							# target
RequiredBy=				# unit被需求的target
Alias=					# unit启动别名
Also=					# `enable`当前unit时，同时被`enable`
							# 的unit
```

-	`Type`：服务启动类型

	-	`simple`：默认，执行`ExecStart`启动主进程
		-	Systemd认为服务立即启动
		-	适合服务在前台持续运行
		-	服务进程不会fork
		-	若服务要启动其他服务，不应应该使用此类型启动，

	-	`forking`：以fork方式从父进程创建子进程，然后父进程
		立刻退出
		-	父进程退出后Systemd才认为服务启动成功
		-	适合常规守护进程（服务在后台运行）
		-	启动同时需要指定`PIDFile=`，以便systemd能跟踪
			服务主进程

	-	`oneshot`：一次性进程
		-	Systemd等待当前服务退出才继续执行
		-	适合只执行一项任务、随后立即退出的服务
		-	可能需要同时设置`RemainAfterExit=yes`，使得
			systemd在服务进程推出后仍然认为服务处于激活状态

	-	`notify`：同`simple`
		-	但服务就绪后向Systemd发送信号

	-	`idle`：其他任务执行完毕才会开始服务

	-	`dbus`：通过D-Bus启动服务
		-	指定的`BusName`出现在DBus系统总线上时，Systemd
			即认为服务就绪

-	`WantedBy`：服务安装的用户模式，即希望使用服务的用户
	-	`multi-user.target`：允许多用户使用服务

####	`.target`

`<some_name>.target`：可以理解为系统的“状态点”

-	target通常包含多个unit
	-	即包含需要启动的服务的组
	-	方便启动一组unit

-	启动target即意味将系统置于某个状态点

-	target可以和传统init启动模式中RunLevel相对应
	-	但RunLevel互斥，不能同时启动
	
	|RunLevel|Target|
	|-----|-----|
	|0|`runlevel0.target`或`poweroff.target`|
	|1|`runlevel1.target`或`rescue.target`|
	|2|`runlevel2.target`或`multi-user.target`|
	|3|`runlevel3.target`或`multi-user.target`|
	|4|`runlevel4.target`或`multi-user.target`|
	|5|`runlevel5.target`或`graphical.target`|
	|6|`runlevel6.target`或`reboot.target`|

-	`enable`设置某个服务自启动，就是在服务配置`.service`中
	`WantedBy`、`RequiredBy`指明的target注册

	-	`WantedBy`向target对应目录中
		`/etc/systemd/system/<target_name>.target.wants`
		添加符号链接
	-	`RequiredBy`向target对应目录中
		`/etc/systemd/system/<target_name>.target.required`
		添加符号链接
	-	`systemctl disable <unit>`就是移除对应symlink

####	`.socket`

###	日志

####	`/etc/systemd/journal.conf`

Systemd日志配置文件

##	网络

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

###	HOST

####	`/etc/hostname`

设置主机名称，直接填写字符串即可

```
PC-NAME
```

####	`/etc/hosts`

ipd地址-主机名称映射

-	理论上说，这里主机名称应该是本机称呼，不要求其他主机
	`/etc/hostname`与这里的主机名一致

```
127.0.0.1 localhost
xxx.xxx.xxx.xxx name
xxx.xxx.xxx.xxx domain.name
```

###	DNS Resolver

####	相关问题

-	`Temporary failure in name resolution`
	-	问题：可能是DNS服务器配置缺失、错误
	-	场景：`$ ping`
	-	解决：设置`nameserver`配置DNS服务器地址

####	`/etc/resolv.conf`

域名解析器（resolver）（DNS客户机）配置文件

-	设置DNS服务器IP地址、DNS域名
-	包含主机域名搜索顺序

```cnf
nameserver		114.114.114.114		# DNS服务器IP地址
domain			localhost			# 本地域名
search			search_list			# 域名搜索列表
sortlist							# 允许将得到域名结果进行排序
```

说明

-	`nameserver`：可以有多**行**，每行一个ip地址，查询时按照
	顺序依次查找
-	`domain`：声明主机域名
	-	查询无域名主机时需要使用
	-	邮件系统需要使用
	-	未配置则使用主机名
-	`search`：其参数指明域名查询顺序
	-	查询无域名主机时，将在其参数声明域中分别查找
	-	`domain`、`search`不共存，同时存在时，后者覆盖前者
-	`sortlist`：对得到的域名结果进行特定排序
	-	参数未网络/掩码对时，允许任意排列顺序

##	硬件

###	磁盘挂载

####	`/etc/fstab`

包含存储设备、文件系统信息，配置**自动挂载**各种文件系统格式
硬盘、分区、可移动设备、远程设备（即`mount`参数存盘）

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

##	系统日志

###	`/var/log`

-	`bootstrap.log`：系统引导相关信息
-	`cron`：系统调度执行信息
-	`dmesg`：内核启动时信息，包括硬件、文件系统
-	`maillog`：邮件服务器信息
-	`message`：系统运行过程相关信息，包括IO、网络
-	`secure`：系统安全信息

##	标准输出

###	文件状态

![ls_results.png](imgs/ls_results.png)

-	文件权限：包括10个字符

	-	第1字符：文件类型
		-	`-`；普通文件
		-	`d`：目录
		-	`l`：link，符号链接
		-	`s`：socket
		-	`b`：block，块设备
		-	`c`：charactor，字符设备（流）
		-	`p`：FIFO Pipe
	-	第2-4字符：owner，文件属主权限
	-	第5-7字符：group，同组用户权限
	-	第8-10字符：other，其他用户权限
		-	权限分别为`r`读、`w`写、`x`执行
		-	相应位置为`-`表示没有此权限
	-	执行位还可能是其他特殊字符
		-	user`s`：文件set-user-id、执行权限同时被置位
		-	group`s`：文件set-group-id、执行权限同时被置位
		-	user`S`：文件set-user-id被置位，执行权限未置位
		-	group`S`：文件set-group-id被置位，执行权限未置位
		-	other`t`：文件sticky bit、执行权限均被置位
		-	other`T`：文件sticky bit被置位、执行权限未置位

	> - 关于权限具体含义，参见`linux/kernel/file_system`
	> - 权限设置，参见`linux/shell/cmd_fds`

-	文件数量

	-	一般文件：硬链接数目
	-	目录：目录中第一级子目录个数

-	文件属主名

-	文件属主默认用户组名

-	文件大小（Byte）

-	最后修改时间

-	文件名

##	系统环境变量

###	NAME

-	`PATH`：用户命令查找目录
-	`HOME`：用户主工作目录
-	`SHELL`：用户使用shell
-	`LOGNAME`：用户登录名
-	`LANG/LANGUAGE`：语言设置
-	`MAIL`：用户邮件存储目录
-	`PS1`：命令基本提示符
-	`PS2`：命令附属提示符
-	`HISTSIZE`：保存历史命令记录条数
-	`HOSTNAME`：主机名称

> - `/etc/passwd`、`/etc/hostname`等文件中设置各用户部分
	默认值，缺省随系统改变

###	PATH

####	C PATH

-	`LIBRARY_PATH`：程序编译时，动态链接库查找路径
-	`LD_LIBRARAY_PATH`：程序加载/运行时，动态链接库查找路径

> - 动态链接库寻找由`/lib/ld.so`实现，缺省包含`/usr/lib`、
	`/usr/lib64`等
> > - 建议使用`/etc/ld.so.conf`配置替代`LD_LIBRARY_PATH`，
	或在编译时使用`-R<path>`指定
> > - 手动添加动态链接库至`/lib`、`/usr/lib`等中时，可能
	需要调用`ldconfig`生成cache，否则无法找到


