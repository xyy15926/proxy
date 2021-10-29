---
title: Linux 网络接口配置
categories:
  - Linux
  - Network
tags:
  - Linux
  - Kernel
  - Network
  - Host
  - DNS
date: 2021-07-29 20:51:30
updated: 2021-07-29 20:52:15
toc: true
mathjax: true
description: 
---

##	网络配置

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

