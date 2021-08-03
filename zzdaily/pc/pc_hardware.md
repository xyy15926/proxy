---
title: PC硬件
tags:
  - Daily Life
categories:
  - Daily Life
  - Hard Drive
  - SSD
  - SATA
  - PCI-E
  - NVMe
  - AHCI
  - M.2
date: 2019-03-21 17:27:37
updated: 2021-07-19 09:17:58
toc: true
mathjax: true
comments: true
description: PC硬件
---

##	SSD硬盘

###	接口/插槽类型

####	`M.2`接口/插槽

`M.2`曾经称为NGFF

-	`M.2`接口的SSD有很多长度版本，常用的是42/60/80mm这三种
	版本

-	`M.2`根据接口（金手指）/插槽的缺口位置、针脚数量可以
	分为两类

	-	`B Key(Socket2)`：缺口偏左，左侧6个针脚宽，对应SSD
		金手指（接口）左侧6个针脚（插槽上针脚数-1为5）

	-	`M Key(Socket3)`：缺口偏右，右侧5个针脚宽，对应SSD
		金手指（接口）右侧5个针脚（插槽上针脚数-1为4）

`B&M`：大部分`M.2`SSD跳过了`B Key`金手指，采用`B&M`类型的
金手指，这种类型的金手指有偏左、偏右均有一个缺口，兼容
`B Key`和`M Key`类型的插槽

####	`SATA`串口

####	`mSATA`串口

####	`IDE`并口

###	总线标准

####	`PCI-E`PCI-Express

-	`PCI-E 3.0*2`：
-	`PCI-E 3.0*4`：总带宽有32Gbps

####	`SATA3.0`

`SATA3.0`带宽仅有6Gbps，采用此类总线标准的SSD速度较慢

	
###	传输协议

####	`AHCI`

`AHCI`是`SATA`总线对应的传输协议标准（逻辑设备接口标准），
可以看作是一种`SATA`的优化驱动

>	可以在BIOS中开启该协议

####	`NVMe` 
`NVMe`是针对`PCI-E`总线SSD的告诉传输协议

就目前而言，支持`NVMe`协议的`M.2`SSD一定采用了`PCI-E 3.0*4`
总线标准，但反之不一定

###	总结

`M.2`SSD

-	`B&M`金手指
	-	`SATA`总线：<600MB/s
	-	`PCI-E 3.0*2`总线：<1000MB/s
-	`M Key`金手指
	-	`PCI-E 3.0*2`总线：<1000MB/s
	-	`PCI-E 3.0*4`总线
		-	支持`NVMe`协议：可以超过2000MB/s
		-	不支持`NVMe`协议：<1500MB/s

