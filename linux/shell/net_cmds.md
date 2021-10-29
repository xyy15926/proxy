---
title: Linux 网络接口命令
categories:
  - Linux
  - Network
tags:
  - Linux
  - Kernel
  - Network
  - Ping
  - Route
date: 2021-07-29 21:43:22
updated: 2021-09-02 09:46:49
toc: true
mathjax: true
description: 
---

##	网络状态

###	`netstat`

`netstat`：显示与网络相关的状态信息

-	可查看信息包括
	-	网络连接状态
	-	接口配置信息、
	-	检查路由表
	-	统计信息

-	参数
	-	`-a`：显示网络所有连接中的scoket
	-	`-c`：持续列出网络状态
	-	`-i`：显示网络界面信息表单
	-	`-n`：直接使用IP地址而不是主机名称
	-	`-N`：显示网络硬件外围设备号连接名称
	-	`-s`：显示网络工作信息统计表
	-	`-t`：显示TCP传输协议连接状况

###	`ping`

-	`ping`：向被测试目的主机地址发送ICMP报文并收取回应报文

-	选项参数
	-	`-c`：要求回应的次数
	-	`-i`：发送 *ICMP* 报文时间间隔
	-	`-R`：记录路由过程
	-	`-s`：数据包大小
	-	`-t`：存活数值（路由跳数限制）

###	`traceroute`

-	`traceroute`：跟踪 *UDP* 路由数据报

-	选项参数
	-	`-g`：设置来源来路由网关
	-	`-n`：直接使用IP地址而不是主机名称
	-	`-p`：设置UDP传输协议的通信端口
	-	`-s`：设置本地主机送出数据包的IP地址
	-	`-w`：超时秒数（等待远程主机回报时间）

##	网络配置

###	`ifconfig`

-	`ifconfig`：显示、设置网络

-	命令参数
	-	`netmask`：设置网卡子网掩码
	-	`up`：启动指定网卡
	-	`down`：关闭指定网络设备
	-	`ip`：指定网卡ip地址

##	命令总结

-	网络状态
	-	`ping`：向被测试目的主机地址发送ICMP报文并收取回应报文
	-	`netstat`：显示与网络相关的状态信息
	-	`route`：查看、配置 *Linux* 系统上的路由信息
	-	`traceroute`：跟踪 *UDP* 路由数据报

-	网络配置
	-	`ifconfig`：显示、设置网络

