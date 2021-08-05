---
title: RFC
categories:
  - CS
  - Network
tags:
  - CS
  - Network
  - OSI
date: 2019-05-01 09:58:40
updated: 2019-05-01 09:58:40
toc: true
mathjax: true
comments: true
description: RFC
---

##	局域网协议

###	*Identification Protocal*

Ident协议

-	如果客户端支持Ident协议，可以在TCP端口113上监听ident请求
	-	基本每个类Unix操作系统都带有Ident协议

-	Ident在组织内部可以很好工作，但是在公共网络不能很好工作
	-	很多PC客户端没有运行Ident识别协议守护进程
	-	Ident协议会使HTTP事务处理产生严重时延
	-	很多防火墙不允许Ident流量进入
	-	Ident协议不安全，容易被伪造
	-	Ident协议不支持虚拟IP地址
	-	暴露客户端用户名涉及隐私问题

-	Ident协议不应用作认证、访问控制协议

##	Internet协议



