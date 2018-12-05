#	Linux特殊文件

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
UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
							# 通用唯一识别码：不能相同（注意虚拟机）

PROXY_METHOD=none			# 代理方式：无
BROWSER_ONLY=no				# 仅浏览器：否

USERCTL=no					# 是否允许非root用户控制此设备：否
MASTER=						# 主设备名称？
SLAVE=						# 从社会名称？
NETWORK=					# 网络地址
```

###	`/etc/sysconfig/grub`

GRUB配置文件，实际上是`/etc/default/grub`的软连接

###	服务

####	`/usr/lib/systemd`

此文件夹包含一系列服务

-	服务文件就是文本文件，包含服务需要执行的程序、描述等

#####	`.service`

#####	`.target`

#####	`.socket`


####	`/etc/systemd/system/`

此文件夹中均为`systemctl enable`的服务symlink

-	其中包含的服务symlink会在机器启动时启动
-	`systemctl enable`就是在`/etc/systemd/system`
	（或其子文件夹）创建服务的symlink
-	`systemctl disable`就是移除symlink

##	网络连通

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
Host link_name
	HostName host
	User user_name
	Port 22
	IdentityFile private_key
		# 默认私钥文件应该是`~/.ssh`下的文件
```

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

	>	关于权限具体含义，参见`shell_puzzles.md`

-	文件数量

	-	一般文件：硬链接数目
	-	目录：目录中第一级子目录个数

-	文件属主名

-	文件属主默认用户组名

-	文件大小（Byte）

-	最后修改时间

-	文件名

